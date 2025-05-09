import copy
from datetime import datetime, timezone
import math
import os
from collections import namedtuple
import random
import pytz

import yaml
import pprint

import gym
import torch.nn as nn
from torchvision.models import VGG16_Weights
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from utils.generate_mmcovs import generate_mmconvs
from utils.get_attentionmap import get_attention_map
from utils.positionalencoding2d import positionalencoding2d
import visual_foraging_gym
import numpy as np
import torch

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision import transforms

from PIL import Image, ImageOps

from utils.models.gen_actor_critic_ablation1 import Actor, TaskEmbedding
from utils.gen_ppo_memory import Memory
from TestLargePPOOutOfDomain import test
from TestLargePPOOutOfDomainShuffle import test as test_shuffle

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", type=int) # 这个参数很重要
# args = parser.parse_args()
# torch.distributed.init_process_group(backend='nccl')

Transition = namedtuple(
    "Transition",
    (
        "attention_map",
        "action",
        "action_logprob",
        "next_attention_map",
        "reward",
        "done",
        "state_value",
        "values",
    ),
)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# load config
with open("utils/config.yml", "r") as file:
    config = yaml.safe_load(file)
pprint.pprint(config)
BATCH_SIZE = config["ppo"]["batch size"]
target_size = 64
clip_eps = config["ppo"]["clip epsilon"]
gamma = config["ppo"]["gamma"]
lam = config["ppo"]["lambda"]
entropy_coefficient = config["hyperparameter"]["entropy coefficient"]
block_num = config["hyperparameter"]["block num"]
head_num = config["hyperparameter"]["head num"]

torch.cuda.set_device(1)
device = torch.device("cuda")
actor = Actor(block_num, head_num)
test_embedding = TaskEmbedding()
test_embedding = test_embedding.to(device)
actor.to(device)
# actor = torch.nn.parallel.DistributedDataParallel(actor, device_ids=[args.local_rank], find_unused_parameters=True)
actor = nn.DataParallel(actor, device_ids=[1])


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)


if config["ppo"]["model init"]:
    actor.apply(init_weights)

episode_saved = 0
CHECK_POINT_PATH = "data/model/ablation-valem.pt"
# ini_weight = torch.load("data/model/attention89.pt")
# episode_saved = ini_weight["episode"]
# actor.load_state_dict(ini_weight["model_state_dict"])
# ini_weight = 0


memory = Memory(config["ppo"]["num_actor"] * (config["ppo"]["max episode step"] + 1))
optimizer = torch.optim.Adam(
    actor.parameters(),
    lr=float(config["hyperparameter"]["learning rate"]),
    weight_decay=config["hyperparameter"]["weight decay"],
)
embedding_optimizer = torch.optim.Adam(
    test_embedding.parameters(),
    lr=float(config["hyperparameter"]["learning rate"]),
    weight_decay=0,
)
criterion = nn.SmoothL1Loss()

# load env config
with open("visual_foraging_gym/envs/env_config.yml", "r") as file:
    env_config = yaml.safe_load(file)

size = env_config["variable"]["size"]
vgg_model = torch.hub.load(
    "pytorch/vision:v0.10.0", "vgg16", weights=VGG16_Weights.DEFAULT
)
model_stimuli = vgg_model.features
model_stimuli = model_stimuli.to(device)

pe = positionalencoding2d(512, size, size)
pe = pe.to(device)

vgg_model = vgg_model.to(device)

env = gym.make("visual_foraging_gym/VisualForaging-v1.5", render_mode="rgb_array")
env.seed(42)
test_env = gym.make(
    "visual_foraging_gym/VisualForaging-v1.6",
    render_mode="rgb_array",
    task_mode=1,
    values=[2, 4, 8, 16],
)
test_env_3 = gym.make(
    "visual_foraging_gym/VisualForaging-v1.6",
    render_mode="rgb_array",
    task_mode=2,
    values=[2, 4, 8, 16],
)
test_env_2 = gym.make(
    "visual_foraging_gym/VisualForaging-v1.6",
    render_mode="rgb_array",
    task_mode=1,
    values=[16, 8, 4, 2],
)
test_env_g = gym.make(
    "visual_foraging_gym/VisualForaging-v1.6",
    render_mode="rgb_array",
    task_mode=1,
    values=[16, 18, 20, 22],
)


def optimize_model():
    if len(memory.memory) < BATCH_SIZE:
        return 0, 0, 0, 0
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # actor.train()

    state_value = torch.tensor(batch.state_value)
    state_value = state_value.detach()
    rewards = batch.reward
    dones = batch.done

    advantages = cal_gae(rewards, state_value, dones)

    # torch.cuda.set_device(args.local_rank)
    attention_map = torch.cat(batch.attention_map)
    attention_map.to(device)
    attention_map = attention_map.detach()
    values = torch.cat(batch.values)
    values = values.detach()
    values = test_embedding(values)
    policy, value = actor(attention_map, values)
    policy = nn.functional.softmax(policy, 1)
    dist = Categorical(policy)

    action = torch.tensor(batch.action, device=device).unsqueeze(0)
    action = action.detach()
    action_logprobs = dist.log_prob(action)

    entropy = dist.entropy().mean()
    # advantages = advantages.to(device) * nn.functional.softmax(entropy, 0) * BATCH_SIZE
    # entropy = entropy.mean()

    action_logprob_old = torch.tensor(batch.action_logprob, device=device)
    action_logprob_old = action_logprob_old.detach()

    ratios = torch.exp(action_logprobs.squeeze() - action_logprob_old)
    approx_kl = (action_logprob_old - action_logprobs).mean()
    advantages = advantages.detach().to(device)
    sur_1 = ratios * advantages
    sur_2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    clip_loss = -torch.min(sur_1, sur_2)
    clip_loss = clip_loss.mean()

    actor_loss = clip_loss - entropy_coefficient * entropy

    if config["ppo"]["monter car"]:
        target_values = value_function_estimation(rewards, dones).detach().to(device)
    else:
        target_values = state_value.to(device) + advantages

    value = value.squeeze()
    # target_values = (target_values - target_values.mean()) / target_values.std()

    critic_loss = criterion(value, target_values)

    if approx_kl > 0.015:
        loss = critic_loss
    else:
        loss = critic_loss + actor_loss

    # Optimize the model
    optimizer.zero_grad()
    embedding_optimizer.zero_grad()
    # actor_loss.backward()
    # critic_loss.backward()
    loss.backward()
    if config["ppo"]["global grad clip"]:
        for param in actor.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
    optimizer.step()
    embedding_optimizer.step()

    # checkpoint = torch.load(CHECK_POINT_PATH)
    # print('actor loss', checkpoint['actor_loss'],
    #       '\ncritic loss', checkpoint['critic_loss'])
    return actor_loss.item(), critic_loss.item(), entropy.item(), approx_kl.item()


def cal_gae(rewards, values, dones):
    T = len(rewards)
    reward_batch = torch.tensor(rewards).float()
    advantages = torch.zeros_like(reward_batch)
    advantage = 0
    next_value = 0

    for t in reversed(range(T)):
        done = dones[t]
        td_error = reward_batch[t] + next_value * (1 - done) * gamma - values[t]
        advantage = td_error + advantage * gamma * lam * (1 - done)
        next_value = copy.copy(values[t])
        advantages[t] = advantage

    if config["ppo"]["advantage normalize"]:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages


def value_function_estimation(rewards, dones):
    T = len(rewards)
    reward_batch = torch.tensor(rewards).float()
    target_values = torch.zeros_like(reward_batch)
    target_value = 0
    for t in reversed(range(T)):
        done = dones[t]
        reward = reward_batch[t]
        target_value = reward + target_value * (1 - done) * gamma
        target_values[t] = target_value

    return target_values.squeeze()


def select_action(actor, test_embedding, attention_map, values):
    with torch.no_grad():
        values = test_embedding(values)
        policy, value = actor(attention_map, values)
        policy = nn.functional.softmax(policy, 1)
        policy = policy.squeeze()
        dist = Categorical(policy)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        action_entropy = dist.entropy()

    return action, action_logprob, value.squeeze(), action_entropy


observation, info = env.reset()
filenames = info["filename"][0:4]
MMconvs, mean, std = generate_mmconvs(filenames, vgg_model)
fixations = [[8, 8]]
order_idx = [0, 1, 2, 3]
random.shuffle(order_idx)
MMconvs_new = []
next_values = []
for o in range(4):
    MMconvs_new.append(MMconvs[order_idx[o]])
    next_values.append(info["value"][order_idx[o]])
attention_map = get_attention_map(
    observation, fixations, model_stimuli, MMconvs_new, env_config, mean, std, pe
)
values = torch.tensor(next_values, device=device).float().unsqueeze(0)

# Save performance
Average_Reward, Average_Value, Average_Entropy, Average_Score = [], [], [], []
# Save Loss
Policy_Loss, Value_Loss = [], []
Approx_kl = []
TestScore, TestScoreG, TestScore2, TestScoreS, TestScore3 = [], [], [], [], []

for i_episode in range(config["ppo"]["num_episode"]):
    memory.clean()

    average_values, action_entropys, average_rewards, average_score = 0, 0, 0, 0
    policy_loss, value_loss = 0, 0

    for i_actor in range(config["ppo"]["num_actor"]):
        average_value, average_reward, action_entropy, step_in_episode = 0, 0, 0, 0

        while True:

            actor.eval()

            action, action_logprob, state_value, entropy = select_action(
                actor, test_embedding, attention_map, values
            )

            b, a = divmod(int(action.cpu()), env_config["variable"]["size"])
            fixations.append([a, b])
            next_observation, reward, terminated, truncated, info = env.step(
                np.array([a, b])
            )

            random.shuffle(order_idx)
            MMconvs_new = []
            next_values = []
            for o in range(4):
                MMconvs_new.append(MMconvs[order_idx[o]])
                next_values.append(info["value"][order_idx[o]])

            next_attention_map = get_attention_map(
                next_observation,
                fixations,
                model_stimuli,
                MMconvs_new,
                env_config,
                mean,
                std,
                pe,
            )
            next_values = torch.tensor(next_values, device=device).float().unsqueeze(0)
            average_reward += reward
            if not config["ppo"]["check env"]:
                average_value += state_value
                action_entropy += entropy

            step_in_episode += 1

            if (
                terminated
                or truncated
                or step_in_episode > config["ppo"]["max episode step"]
            ):
                next_attention_map = None

            memory.push(
                attention_map.cpu(),
                action,
                action_logprob,
                next_attention_map,
                reward,
                terminated or step_in_episode > config["ppo"]["max episode step"],
                state_value,
                values,
            )

            attention_map = copy.deepcopy(next_attention_map)
            values = copy.deepcopy(next_values)
            if (
                terminated
                or truncated
                or step_in_episode > config["ppo"]["max episode step"]
            ):
                average_score += env.SCORE

                observation, info = env.reset()
                filenames = info["filename"][0:4]
                MMconvs, mean, std = generate_mmconvs(filenames, vgg_model)
                # MMconvs, mean, std = generate_mmconvs(info['target images'])
                fixations = [[8, 8]]
                random.shuffle(order_idx)
                MMconvs_new = []
                next_values = []
                for o in range(4):
                    MMconvs_new.append(MMconvs[order_idx[o]])
                    next_values.append(info["value"][order_idx[o]])
                attention_map = get_attention_map(
                    observation,
                    fixations,
                    model_stimuli,
                    MMconvs_new,
                    env_config,
                    mean,
                    std,
                    pe,
                )
                values = torch.tensor(next_values, device=device).float().unsqueeze(0)
                average_reward = average_reward / step_in_episode
                average_value = average_value / step_in_episode
                action_entropy = action_entropy / step_in_episode
                break
        average_values += average_value
        action_entropys += action_entropy
        average_rewards += average_reward

    print(
        i_episode + episode_saved,
        "average_score",
        average_score / config["ppo"]["num_actor"],
        "average_values",
        average_values.item() / config["ppo"]["num_actor"],
        "action entropy",
        action_entropys.item() / config["ppo"]["num_actor"],
    )
    Average_Reward.append(average_rewards / config["ppo"]["num_actor"])
    Average_Value.append(average_values.item() / config["ppo"]["num_actor"])
    Average_Entropy.append(action_entropys.item() / config["ppo"]["num_actor"])
    Average_Score.append(average_score / config["ppo"]["num_actor"])

    kl = 0
    for i_epoch in range(config["ppo"]["num epoch"]):
        actor_loss, critic_loss, entropy, approx_kl = optimize_model()
        policy_loss += actor_loss
        value_loss += critic_loss
        kl = kl + approx_kl

    # entropy_coefficient = entropy_coefficient * 0.99999

    Policy_Loss.append(policy_loss / config["ppo"]["num epoch"])
    Value_Loss.append(value_loss / config["ppo"]["num epoch"])
    Approx_kl.append(kl / config["ppo"]["num epoch"])

    if config["ppo"]["save"]:
        torch.save(
            {
                "episode": i_episode + episode_saved,
                "model_state_dict": actor.module.state_dict(),
                "embedding_model_state_dict": test_embedding.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "average reward": Average_Reward,
                "average value": Average_Value,
                "average entropy": Average_Entropy,
                "average score": Average_Score,
                "policy loss": Policy_Loss,
                "value loss": Value_Loss,
                "kl divergence": Approx_kl,
                "test score": TestScore,
                "test score 2": TestScore2,
                "test score 3": TestScore3,
                "test score gen": TestScoreG,
                "test score shuffle": TestScoreS,
            },
            CHECK_POINT_PATH,
        )

    _, test_score, _, _ = test(CHECK_POINT_PATH, test_env, vgg_model)
    TestScore.append(test_score[0])

    _, test_score2, _, _ = test(CHECK_POINT_PATH, test_env_2, vgg_model)
    TestScore2.append(test_score2[0])

    _, test_score3, _, _ = test(CHECK_POINT_PATH, test_env_3, vgg_model)
    TestScore3.append(test_score3[0])

    _, test_score_g, _, _ = test(CHECK_POINT_PATH, test_env_g, vgg_model)
    TestScoreG.append(test_score_g[0])
    _, test_score_s, _, _ = test_shuffle(CHECK_POINT_PATH, test_env, vgg_model)
    TestScoreS.append(test_score_s[0])

sgt = pytz.timezone("Asia/Singapore")

current_time_singapore = datetime.now(sgt)

# Format the time and print it
formatted_time_singapore = current_time_singapore.strftime("%Y-%m-%d %H:%M:%S %Z")
print("Done!", f"finish time: {formatted_time_singapore}")
