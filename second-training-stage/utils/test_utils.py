import os
import csv
import random
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical


def read_stimulus(directory, name, is_header=False):
    data = []
    file_path = os.path.join(directory, name)
    with open(file_path) as f:
        reader = csv.reader(f)
        if is_header:
            header = next(reader)
        for row in reader:
            data.append(row)
    return data


def prepare_stimuli(stimuli_id, filenames, distractorIndex, itemPositions, popularities, values):
    target_image_file_list = filenames[stimuli_id][0:4]
    distractor_image_file_list = filenames[stimuli_id][4:13]
    distractor = distractorIndex[stimuli_id]
    distractor_index = []
    for d in distractor:
        if d != 'NaN':
            d = int(d)
            distractor_index.append(d-1)
    all_sprited = itemPositions[2*stimuli_id:2*(stimuli_id+1)]
    all_sprited_positions = []
    for i in range(120):
        if all_sprited[0][i] != 'NaN':
            position = np.array(
                [int(all_sprited[0][i]), int(all_sprited[1][i])])
            all_sprited_positions.append(position)
    popularity = []
    for p in popularities[stimuli_id]:
        popularity.append(int(p))
    value = []
    for v in values[stimuli_id]:
        value.append(int(v))
    return target_image_file_list, distractor_image_file_list, distractor_index, all_sprited_positions, popularity, value


def generate_ood7_stimuli(stimuli_id, popularities, values):
    # Target images
    folder_path = 'visual_foraging_gym/envs/TargetFixed'
    all_files = os.listdir(folder_path)
    target_image_file_list = [
        file for file in all_files if file.lower().endswith(('.jpg'))]
    random.shuffle(target_image_file_list)
    target_image_file_list = target_image_file_list[0:1]
    target_image_file_list = target_image_file_list * 4
    # Distractor images
    folder_path = 'visual_foraging_gym/envs/DistractorFixed'
    all_files = os.listdir(folder_path)
    distractor_image_file_list = [
        file for file in all_files if file.lower().endswith(('.jpg'))]
    # Distractor permutation
    distractor_index = random.choices(range(9), k=(105-32))
    # Item position
    positions = set()
    while len(positions) < 105:
        row = random.randint(0, 16 - 1)
        column = random.randint(0, 16 - 1)
        positions.add((row, column))
    positions = list(positions)
    all_sprited_positions = [np.array(position) for position in positions]
    popularity = []
    for p in popularities[stimuli_id]:
        popularity.append(int(p))
    value = []
    for v in values[stimuli_id]:
        value.append(int(v))
    return target_image_file_list, distractor_image_file_list, distractor_index, all_sprited_positions, popularity, value


def select_action(actor, task_embedding, attention_map, value):
    value = task_embedding(value)
    policy, value = actor(attention_map, value)
    policy = nn.functional.softmax(policy, 1)
    policy = policy.squeeze()
    dist = Categorical(policy)

    action = dist.sample()
    action_logprob = dist.log_prob(action)
    action_entropy = dist.entropy()

    return action, action_logprob, value.squeeze(), action_entropy
