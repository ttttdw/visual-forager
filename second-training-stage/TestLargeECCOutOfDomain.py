import os
import gym
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from utils.ecc_net import load_eccNet
import visual_foraging_gym
from utils.models.gen_actor_critic19 import Actor
from torch.distributions import Categorical
from torchvision.models import VGG16_Weights
from utils.positionalencoding2d import positionalencoding2d
from utils.get_attentionmap import get_eccattention_map
from utils.generate_mmcovs import generate_mmconvs


def select_action(actor, attention_map, value):
    policy, value = actor(attention_map, value)
    policy = nn.functional.softmax(policy, 1)
    policy = policy.squeeze()
    dist = Categorical(policy)

    action = dist.sample()
    action_logprob = dist.log_prob(action)
    action_entropy = dist.entropy()

    return action, action_logprob, value.squeeze(), action_entropy


def test(path, env, vgg_model, device):
    # device = torch.device('cuda')
    actor = Actor(12, 4)
    CHECK_POINT_PATH = (
        path
    )
    checkpoint = torch.load(CHECK_POINT_PATH)
    actor.load_state_dict(checkpoint["model_state_dict"])
    actor.to(device=device)
    # load env config
    with open("visual_foraging_gym/envs/env_config.yml", "r") as file:
        env_config = yaml.safe_load(file)
    size = env_config["variable"]["size"]
    target_size = env_config["variable"]["target size"]
    model_stimuli = load_eccNet(
        (
            1,
            3,
            target_size * (size * 2 - 1),
            target_size * (size * 2 - 1),
        )
    ).to(device)
    # vgg_model = torch.hub.load(
    # "pytorch/vision:v0.10.0", "vgg16", weights=VGG16_Weights.DEFAULT)
    vgg_model = vgg_model.to(device)
    pe = positionalencoding2d(512, size, size)
    pe = pe.to(device)
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    # env = gym.make("visual_foraging_gym/VisualForaging-v1.1",
    #                render_mode="rgb_array", task_mode=task_mode)
    observation, info = env.reset()
    filenames = info['filename'][0:4]
    MMconvs, mean, std = generate_mmconvs(filenames, vgg_model)
    # obs_img = transforms.ToPILImage()(observation)
    # obs_img.save("obs_img.jpg")
    fixations = [[8, 8]]
    attention_map = get_eccattention_map(
        observation, fixations, model_stimuli, MMconvs, env_config, mean, std, pe
    )
    step, episode, score = 0, 0, 0
    saccades = []
    scores = []
    click_count = {'click_one': [],
                   'click_two': [],
                   'click_three': [],
                   'click_four': [],
                   'click_distractor': []
                   }
    click_observer = np.zeros((20, 5))
    for _ in range(20):
        values = torch.tensor(info["value"]).float().unsqueeze(0).to(device)
        action, _, _, _ = select_action(actor, attention_map, values)
        b, a = divmod(int(action.cpu()), size)
        saccades.append(
            [a-fixations[-1][0], b-fixations[-1][1]])
        fixations.append([a, b])
        observation, reward, terminated, truncated, info = env.step(
            np.array([a, b]))
        attention_map = get_eccattention_map(
            observation, fixations, model_stimuli, MMconvs, env_config, mean, std, pe
        )
        now_click = info['now click']
        if not now_click is None:
            click_observer[step, now_click] += 1
        step += 1

        if step > 19:
            truncated = True

        if terminated or truncated:
            episode += 1
            score += env.SCORE
            scores.append(env.SCORE)
            # print(env.SCORE, step)
            click_count['click_one'].append(
                info['click_count']['click_target_one'])
            click_count['click_two'].append(
                info['click_count']['click_target_two'])
            click_count['click_three'].append(
                info['click_count']['click_target_three'])
            click_count['click_four'].append(
                info['click_count']['click_target_four'])
            click_count['click_distractor'].append(
                info['click_count']['click_distractor'])
            step = 0
            observation, info = env.reset()
            filenames = info['filename'][0:4]
            MMconvs, mean, std = generate_mmconvs(filenames, vgg_model)
            fixations = [[8, 8]]
            attention_map = get_eccattention_map(
                observation, fixations, model_stimuli, MMconvs, env_config, mean, std, pe
            )
    env.close()
    # print('score', score / episode)
    return saccades, scores, click_count, click_observer / episode


if __name__ == '__main__':
    import json

    path = "data/model/task3ecc.pt"
    saccades, scores, click_count, click_observer = test(path, 2)
    data = {'saccades': saccades, 'scores': scores,
            'click count': click_count, 'click observer': click_observer.tolist()}
    with open('data/test/task3ecc.json', 'w') as json_file:
        json.dump(data, json_file)
