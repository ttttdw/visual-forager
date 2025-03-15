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
import yaml
from core.agent import AnotherFixationAgent, FixationAgent
from core.eye import PerfectEye
from utils.ecc_net import load_eccNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    default="data/model/fine-tune/iormasked-nextclick.pt")
parser.add_argument("--savename", type=str,
                    default='fmodeldatas_iterative.npy')
parser.add_argument("--onlyID1", action='store_true')
parser.add_argument("--onlyID2", action='store_true')
parser.add_argument("--onlyOOD1", action='store_true')
parser.add_argument("--onecondition", action='store_true',
                    help='Test on one condition')
parser.add_argument("--conditionname", type=str)
parser.add_argument("--ecc_mode", action='store_true', help='Enable eccNet')
args = parser.parse_args()

# args.onecondition = True
# args.conditionname = 'condition1'
# args.modelpath = "data/model/fixation-iterative-freeze-penalty-50.pt"
# args.iterative = True
# args.freeze = True

# decision model
# torch.cuda.set_device(2)
from utils.models.anotherFixationModel import Actor, TaskEmbedding
actor = Actor(12, 4)
seed_everything()

actor.apply(init_weights)
task_embedding = TaskEmbedding()
modelpath = args.modelpath
checkpoint = torch.load(modelpath)
actor.load_state_dict(checkpoint["model_state_dict"])
modelpath = "data/model/fixed-image-ppo-3.pt"
checkpoint = torch.load(modelpath)
task_embedding.load_state_dict(checkpoint["embedding_model_state_dict"])

# eye
vgg_model = torch.hub.load(
    "pytorch/vision:v0.10.0", "vgg16", weights=VGG16_Weights.DEFAULT
)
if args.ecc_mode:
    model_stimuli = load_eccNet(
        (
            1,
            3,
            64 * (16 * 2 - 1),
            64 * (16 * 2 - 1),
        )
    )
else:
    model_stimuli = vgg_model.features
# model_stimuli = vgg_model.features
device = torch.device("cuda")
with open('visual_foraging_gym/envs/env_config.yml', 'r') as file:
    env_config = yaml.safe_load(file)
eye = PerfectEye(vgg_model, model_stimuli, device, env_config, ecc_mode=args.ecc_mode)

# env
env = gym.make("visual_foraging_gym/VisualForaging-v2.2",
               render_mode="rgb_array")

# env args


class EnvArgs():
    def __init__(self, size) -> None:
        self.size = size


env_args = EnvArgs(env_config["variable"]["size"])

# initial agent
agent = AnotherFixationAgent(
        actor, task_embedding, eye, env, device, env_args)

# env args


class EnvArgs():
    def __init__(self, target_image_file_list=None, distractor_image_file_list=None, distractor_index=None, all_sprited_positions=None, points=None, popularity=None) -> None:
        self.target_image_file_list = target_image_file_list
        self.distractor_image_file_list = distractor_image_file_list
        self.distractor_index = distractor_index
        self.all_sprited_positions = all_sprited_positions
        self.points = points
        self.popularity = popularity


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
        result = agent.evaluate_on_humanstimulus(
            env_args)
        print(stimuli_id)
        score.append(result['score'])
        fixation_positions.append(result['fixation positions'])
        click_positions.append(result['click positions'])
        cumulative_scores.append(result['cumulative score'])
        click_ratio.append(result['click ratio'])
        click_count[:, :, stimuli_id] = result['click count']
        click_distribution[:, stimuli_id] = result['item percentage']
        radius_score = radius_score + result['radius score']

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
