from itertools import count
import os
from torch import optim
from core.logger import Logger
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import VGG16_Weights
from torch import nn as nn
from utils.dqn_memory import ReplayMemory
from utils.ecc_net import load_eccNet
from utils.learn_utils import seed_everything, init_weights
from utils.models.dqn import DQN
import random
import math
import torch
import copy
import numpy as np
from collections import namedtuple
import visual_foraging_gym
import gym
from torchvision import transforms
from PIL import Image

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))
State = namedtuple('State',
                   ('search_image', 'target_image1', 'target_image2', 'target_image3', 'target_image4', 'points'))

TAU = 0.005
BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100000
LR = 1e-4

seed_everything(42)
device = torch.device("cuda")
memory = ReplayMemory(10000)
steps_done = 0
policy_net = DQN().to(device)
policy_net.apply(init_weights)
# for param in policy_net.features.parameters():
#         param.requires_grad = False
# for param in policy_net.ecc_features.parameters():
#     param.requires_grad = False
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
writer = SummaryWriter("data/logger/dqn-pilot")

policy_net.eval()
target_net.eval()

num_episodes = 5000
episode_step = 200

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.search_image.to(device), state.target_image1.to(device), state.target_image2.to(device), state.target_image3.to(device), state.target_image4.to(device), state.points.to(device)).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randint(0, 511)]])

def select_action_test(state):
    with torch.no_grad():
        return policy_net(state.search_image.to(device), state.target_image1.to(device), state.target_image2.to(device), state.target_image3.to(device), state.target_image4.to(device), state.points.to(device)).max(1).indices.view(1, 1)
    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 0, 0
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool, device=device)
    non_final_next_input1 = torch.cat([s.search_image for s in batch.next_state
                                                if s is not None]).to(device)
    non_final_next_input2 = torch.cat([s.target_image1 for s in batch.next_state
                                                if s is not None]).to(device)
    non_final_next_input3 = torch.cat([s.target_image2 for s in batch.next_state
                                                if s is not None]).to(device)
    non_final_next_input4 = torch.cat([s.target_image3 for s in batch.next_state
                                                if s is not None]).to(device)
    non_final_next_input5 = torch.cat([s.target_image4 for s in batch.next_state
                                                if s is not None]).to(device)
    non_final_next_input6 = torch.cat([s.points for s in batch.next_state
                                                if s is not None]).to(device)
    
    input1_batch = torch.cat([s.search_image for s in batch.state]).to(device)
    input2_batch = torch.cat([s.target_image1 for s in batch.state]).to(device)
    input3_batch = torch.cat([s.target_image2 for s in batch.state]).to(device)
    input4_batch = torch.cat([s.target_image3 for s in batch.state]).to(device)
    input5_batch = torch.cat([s.target_image4 for s in batch.state]).to(device)
    input6_batch = torch.cat([s.points for s in batch.state]).to(device)

    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_action_values = policy_net(input1_batch, input2_batch, input3_batch, input4_batch, input5_batch, input6_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_q_values = policy_net(non_final_next_input1, non_final_next_input2, non_final_next_input3, non_final_next_input4, non_final_next_input5, non_final_next_input6).detach()
        next_actions = next_state_q_values.argmax(dim=1, keepdim=True)
        # next_state_values[non_final_mask] = target_net(non_final_next_input1, non_final_next_input2, non_final_next_input3, non_final_next_input4, non_final_next_input5, non_final_next_input6).max(1).values
        next_state_values[non_final_mask] = target_net(non_final_next_input1, non_final_next_input2, non_final_next_input3, non_final_next_input4, non_final_next_input5, non_final_next_input6).gather(1, next_actions).squeeze()
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    print(mean_grad(policy_net))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1)
    optimizer.step()

    return loss.item(), torch.mean(expected_state_action_values).item()

def mean_grad(model):
    total_grad = 0
    count = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad += param.grad.mean()
            count += 1
    return total_grad / count if count > 0 else 0

def generate_ior_map(fixations, size, fogetting=0.8):
    if len(fixations) == 1:
        return torch.ones((3, size*64, size*64)).to(device)
    ior_map = torch.ones((3, size*64, size*64))
    revers_ior = torch.zeros((3, size*64, size*64))
    for f in fixations[:-1]:
        revers_ior = revers_ior * fogetting
        x_ = int(f[1] * 64)
        x = int((f[1]+1) * 64)
        y_ = int(f[0] * 64)
        y = int((f[0]+1) * 64)
        revers_ior[:, x_:x, y_:y] = 1
        # ior_map[0, f[1], f[0]] = 0
    ior_map = ior_map - revers_ior
    return ior_map.to(device)

def get_state(search_image, info, fixations):
    fixation = fixations[-1]
    ior_map = generate_ior_map(fixations, 16)
    search_image = search_image.float().permute(2, 0, 1) / 255
    search_image = search_image * ior_map
    paddings = (
        (size - fixation[0] - 1) * target_size,
        fixation[0] * target_size,
        (size - fixation[1] - 1) * target_size,
        fixation[1] * target_size,
    )
    search_image = torch.nn.functional.pad(
        search_image, paddings, "constant", 0.5)
    with torch.no_grad():
        search_image = model_stimuli(search_image)
        search_image = search_image[
            :,
            2 * (size - fixation[1] - 1): 2 * (2 * size - fixation[1] - 1),
            2 * (size - fixation[0] - 1): 2 * (2 * size - fixation[0] - 1),
        ]
        search_image = search_image.unsqueeze(0)
        target = []
        for filename in info['filename'][0:4]:
            target_image = Image.open(os.path.join(dir, filename))
            target_image = transform(target_image)
            target_image = model_target(target_image.to(device))
            target.append(target_image.unsqueeze(0).cpu())
    points = torch.tensor(info['value']).float().unsqueeze(0)
    state = State(search_image.cpu(), target[0], target[1], target[2], target[3], points)
    return state
# env
env = gym.make('visual_foraging_gym/VisualForaging-v1.8',
               penalty=-1, fixation_bonus=0)
env.seed(42)
test_env = gym.make("visual_foraging_gym/VisualForaging-v1.9",
                    penalty=-10, fixation_bonus=0)

size = 16
target_size = 64
dir = "visual_foraging_gym/envs/OBJECTSALL"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.8087, std=0.2568)
    ])
logger = Logger()
model_stimuli = load_eccNet(
    (
        1,
        3,
        64 * (16 * 2 - 1),
        64 * (16 * 2 - 1),
    )
).to(device)
vgg_model = torch.hub.load(
    "pytorch/vision:v0.10.0", "vgg16", weights=VGG16_Weights.DEFAULT
)
model_target = vgg_model.features.to(device)

for i_episode in range(num_episodes):
    search_image, info = env.reset()
    fixations = [[8, 8]]
    state = get_state(search_image, info, fixations)
    
    score = []
    Loss = []
    for t in count():
        action = select_action(state)
        click, attention = divmod(int(action.cpu()), 256)
        b, a = divmod(attention, 16)
        fixations.append([a, b])

        # replace this part with env.step
        search_image, reward, terminated, truncated, info = env.step(np.array([click, a, b]))
        observation = get_state(search_image, info, fixations)
        done = terminated or truncated

        reward = torch.tensor([reward])

        if done: #or t == (episode_step-1):
            next_state = None
        else:
            next_state = copy.deepcopy(observation)

        memory.push(state, action.cpu(), next_state, reward)

        state = next_state

        loss, average_q_value = optimize_model()
        Loss.append(loss)
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            score.append(env.SCORE)
            break
            # search_image, info = env.reset()
            # fixation = [8, 8]
            # state = get_state(search_image, info, fixation)
    
    # loss, average_q_value = optimize_model()
    print(i_episode, 'average_score', np.array(score).mean())

    # target_net_state_dict = target_net.state_dict()
    # policy_net_state_dict = policy_net.state_dict()
    # for key in policy_net_state_dict:
    #     target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    # target_net.load_state_dict(target_net_state_dict)

    # search_image, info = test_env.reset()
    # fixation = [8, 8]
    # state = get_state(search_image, info, fixation)
    # for t in count():
    #     action = select_action_test(state)
    #     click, attention = divmod(int(action.cpu()), 256)
    #     b, a = divmod(attention, 16)
    #     fixation = [a, b]

    #     # replace this part with env.step
    #     search_image, reward, terminated, truncated, info = test_env.step(np.array([click, a, b]))
    #     state = get_state(search_image, info, fixation)
    #     done = terminated or truncated

    #     if done:
    #         test_score = test_env.SCORE
    #         break
    
    summary = {'Loss/loss': np.mean(Loss), 'Loss/q_value':average_q_value, 'Score/explore': np.array(score).mean()}
    for key, value in summary.items():
        writer.add_scalar(key, value, i_episode)    
    summary = {'Loss': loss, 'Score/explore': np.array(score).mean()}
    logger.write(summary)

    torch.save({
                'model_state_dict': policy_net.state_dict(),
                'logger': logger.summary
            }, "data/model/dqn-pilot.pt")
