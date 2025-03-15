import scipy.io as sio
import random
from torch import nn as nn
import numpy as np
from utils.test_utils import read_stimulus, prepare_stimuli
from utils.learn_utils import init_weights, seed_everything
import os
import visual_foraging_gym
import gym
import torch
from torchvision.models import VGG16_Weights
from utils.models.dqn import DQN
import argparse
from itertools import count
import os
from core.logger import Logger
from torchvision.models import VGG16_Weights
from torch import nn as nn
from utils.ecc_net import load_eccNet
from utils.learn_utils import seed_everything, init_weights
from utils.models.dqn import DQN
import random
import torch
import numpy as np
from collections import namedtuple
import visual_foraging_gym
import gym
from torchvision import transforms
from PIL import Image

State = namedtuple('State',
                   ('search_image', 'target_image1', 'target_image2', 'target_image3', 'target_image4', 'points'))

parser = argparse.ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    default="data/model/fixation-unified-rewardsearch-10_0.1.pt")
parser.add_argument("--savename", type=str,
                    default='fmodeldatas_iterative.npy')
parser.add_argument("--onlyID1", action='store_true')
parser.add_argument("--onlyID2", action='store_true')
parser.add_argument("--onlyOOD1", action='store_true')
parser.add_argument("--onecondition", action='store_true',
                    help='Test on one condition')
parser.add_argument("--conditionname", type=str)
args = parser.parse_args()

# args.onecondition = True
# args.conditionname = 'condition1'
# args.modelpath = "data/model/fixation-iterative-freeze-penalty-50.pt"
# args.iterative = True
# args.freeze = True

# decision model
seed_everything()
device = torch.device("cuda")
policy_net = DQN().to(device)
modelpath = args.modelpath
checkpoint = torch.load(modelpath)
policy_net.load_state_dict(checkpoint["model_state_dict"])
# env
env = gym.make("visual_foraging_gym/VisualForaging-v2.1",
               render_mode="rgb_array")

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


class EnvArgs():
    def __init__(self, target_image_file_list=None, distractor_image_file_list=None, distractor_index=None, all_sprited_positions=None, points=None, popularity=None) -> None:
        self.target_image_file_list = target_image_file_list
        self.distractor_image_file_list = distractor_image_file_list
        self.distractor_index = distractor_index
        self.all_sprited_positions = all_sprited_positions
        self.points = points
        self.popularity = popularity

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

def get_state(search_image, info, fixations, size=16, target_size=64):
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

def select_action_test(state):
    with torch.no_grad():
        return policy_net(state.search_image.to(device), state.target_image1.to(device), state.target_image2.to(device), state.target_image3.to(device), state.target_image4.to(device), state.points.to(device)).max(1).indices.view(1, 1)
    
def test_one_condition(condition_name):
    directory = os.path.join('data/HS', condition_name)
    image_filenames = read_stimulus(directory, 'filenames.csv', True)
    values = read_stimulus(directory, 'values.csv')
    popularities = read_stimulus(directory, 'popularity.csv')
    itemPositions = read_stimulus(directory, 'itemPositions.csv')
    distractorIndex = read_stimulus(directory, 'distractorIndex.csv')
    score = []
    click_ratio = []
    fixation_positions = []
    click_positions = []
    cumulative_scores = []
    radius_score = []
    click_count = np.zeros((4, 20, len(image_filenames)))
    onscreen_count = np.zeros((4, 20, len(image_filenames)))
    click_distribution = np.zeros((6, len(image_filenames)))

    for stimuli_id in range(len(image_filenames)):
        target_image_file_list, distractor_image_file_list, distractor_index, all_sprited_positions, popularity, points = prepare_stimuli(
            stimuli_id, image_filenames, distractorIndex, itemPositions, popularities, values)
        for item, p in enumerate(popularity):
            onscreen_count[item, :, stimuli_id] = p
        env_args = EnvArgs(target_image_file_list, distractor_image_file_list,
                           distractor_index, all_sprited_positions, points, popularity)
        search_image, info = env.reset(target_image_file_list=env_args.target_image_file_list, distractor_image_file_list=env_args.distractor_image_file_list,
                                           distractor_index=env_args.distractor_index, all_sprited_positions=env_args.all_sprited_positions, values=env_args.points, popularities=env_args.popularity)

        fixations = [[8, 8]]
        cumulative_score = []
        state = get_state(search_image, info, fixations)
        for t in count():
            action = select_action_test(state)
            click, attention = divmod(int(action.cpu()), 256)
            b, a = divmod(attention, 16)
            fixations.append([a, b])

            # replace this part with env.step
            search_image, reward, terminated, truncated, info = env.step(np.array([click, a, b]))
            if click:
                cumulative_score.append(env.SCORE)
            state = get_state(search_image, info, fixations)
            done = terminated or truncated

            if done:
                score = env.SCORE / env.upperbound
                break
        print(stimuli_id, score)    
        cumulative_scores.append(cumulative_score)

    onscreen_count = onscreen_count - np.cumsum(click_count, axis=1)
    click_count = np.sum(click_count, axis=2)
    onscreen_count = np.sum(onscreen_count, axis=2)
    click_percentage = click_count / np.sum(click_count, axis=0)
    onscreen_percentage = onscreen_count / np.sum(onscreen_count, axis=0)

    return {
        'click ratio': np.array(click_ratio),
        'score': np.array(score),
        'click percentage': click_percentage,
        'onscreen percentage': onscreen_percentage,
        'click distribution': click_distribution,
        'fixation positions': fixation_positions,
        'click positions': click_positions,
        'cumulative score': cumulative_scores,
        'radius score': radius_score
    }

# r = test_one_condition('ood8')
# r['score']


conditions = ['condition2', 'condition1', 'condition3',
              'ood2', 'ood3', 'ood4', 'ood5', 'ood7', 'ood6', 'ood8']
if args.onlyID1:
    conditions = ['condition2']
if args.onlyID2:
    conditions = ['condition3']
if args.onlyOOD1:
    conditions = ['condition1']
if args.onecondition:
    conditions = [args.conditionname]
score = {}
click_percentage = {}
onscreen_percentage = {}
for condition_name in conditions:
    print(condition_name)
    r = test_one_condition(condition_name)
    if args.onlyID1 or args.onlyID2 or args.onlyOOD1 or args.onecondition:
        np.save(args.savename, r)
        # sio.savemat(args.savename, r)
        print(r)
    else:
        score[condition_name] = r['score']
        # click_percentage[condition_name] = r['click percentage']
        # onscreen_percentage[condition_name] = r['onscreen percentage']
        sio.savemat(args.savename, score)
