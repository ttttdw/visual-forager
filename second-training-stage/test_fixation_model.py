import numpy as np
import torch
import yaml
from utils.ecc_net import load_eccNet
from utils.generate_mmcovs import generate_mmconvs
from utils.get_attentionmap import get_eccattention_map
from utils.positionalencoding2d import positionalencoding2d
from utils.learn_utils import another_select_action


def test(actor, env, vgg_model, device):
    actor.eval()
    with open('visual_foraging_gym/envs/env_config.yml', 'r') as file:
        env_config = yaml.safe_load(file)
    model_stimuli = load_eccNet(
        (
            1,
            3,
            64 * (16 * 2 - 1),
            64 * (16 * 2 - 1),
        )
    ).to(device)
    pe = positionalencoding2d(512, 16, 16)
    pe = pe.to(device)

    observation, info = env.reset()

    fixations = [[8, 8]]
    filenames = info['filename'][0:4]
    MMconvs, mean, std = generate_mmconvs(filenames, vgg_model)
    values = torch.tensor(info['value'], device=device).float().unsqueeze(0)
    clicked = 1e-4
    correct_click = 1e-4
    missed = 1e-4
    finded = 1e-4
    for step in range(80):
        attention_map = get_eccattention_map(
            observation, fixations, model_stimuli, MMconvs, env_config, mean, std, pe
        )
        action, action_logprob, state_value, entropy = select_action(
            actor, attention_map, values)
        click = action[0]
        clicked += click
        b, a = divmod(int(action[1]),
                      env_config['variable']['size'])
        fixations.append([a, b])
        observation, reward, terminated, truncated, info = env.step(
            np.array([click, a, b]))
        # observation, reward, terminated, truncated, info = env.step(
        #     np.array([click, a, b]))
        if click and reward > 0:
            correct_click += 1
        if reward > 0 and not click:
            missed += 1
        if reward > 0:
            finded += 1
        if terminated or truncated or clicked > 20:
            break
    score = env.SCORE
    print('setp:', step, 'click:', clicked, 'score',
          score, 'correct rate:', correct_click/clicked, 'miss rate:', missed/finded, 'found:', finded)
    return score

def another_test(actor, task_embedding, env, vgg_model, device):
    # split task embedding and ViT
    actor.eval()
    with open('visual_foraging_gym/envs/env_config.yml', 'r') as file:
        env_config = yaml.safe_load(file)
    model_stimuli = load_eccNet(
        (
            1,
            3,
            64 * (16 * 2 - 1),
            64 * (16 * 2 - 1),
        )
    ).to(device)
    pe = positionalencoding2d(512, 16, 16)
    pe = pe.to(device)

    observation, info = env.reset()

    fixations = [[8, 8]]
    filenames = info['filename'][0:4]
    MMconvs, mean, std = generate_mmconvs(filenames, vgg_model)
    values = torch.tensor(info['value'], device=device).float().unsqueeze(0)
    clicked = 1e-4
    correct_click = 1e-4
    missed = 1e-4
    finded = 1e-4
    for step in range(80):
        attention_map = get_eccattention_map(
            observation, fixations, model_stimuli, MMconvs, env_config, mean, std, pe
        )
        action, action_logprob, state_value, entropy = another_select_action(
            actor, task_embedding, attention_map, values)
        click = action[0]
        clicked += click
        b, a = divmod(int(action[1]),
                      env_config['variable']['size'])
        fixations.append([a, b])
        observation, reward, terminated, truncated, info = env.step(
            np.array([click, a, b]))
        # observation, reward, terminated, truncated, info = env.step(
        #     np.array([click, a, b]))
        if click and reward > 0:
            correct_click += 1
        if reward > 0 and not click:
            missed += 1
        if reward > 0:
            finded += 1
        if terminated or truncated or clicked > 20:
            break
    score = env.SCORE
    print('setp:', step, 'click:', clicked, 'score',
          score, 'correct rate:', correct_click/clicked, 'miss rate:', missed/finded, 'found:', finded)
    return score

def test_ood(actor, env, vgg_model, device):
    actor.eval()
    with open('visual_foraging_gym/envs/env_config.yml', 'r') as file:
        env_config = yaml.safe_load(file)
    model_stimuli = load_eccNet(
        (
            1,
            3,
            64 * (16 * 2 - 1),
            64 * (16 * 2 - 1),
        )
    ).to(device)
    pe = positionalencoding2d(512, 16, 16)
    pe = pe.to(device)

    observation, info = env.reset()

    fixations = [[8, 8]]
    filenames = info['filename'][0:4]
    MMconvs, mean, std = generate_mmconvs(filenames, vgg_model)
    values = torch.tensor(info['value'], device=device).float().unsqueeze(0)
    clicked = 1e-4
    correct_click = 1e-4
    missed = 1e-4
    finded = 1e-4
    for step in range(80):
        attention_map = get_eccattention_map(
            observation, fixations, model_stimuli, MMconvs, env_config, mean, std, pe
        )
        action, action_logprob, state_value, entropy = select_action(
            actor, attention_map, values)
        click = action[0]
        clicked += click
        b, a = divmod(int(action[1]),
                      env_config['variable']['size'])
        fixations.append([a, b])
        observation, reward, terminated, truncated, info = env.step(
            np.array([click, a, b]))
        # observation, reward, terminated, truncated, info = env.step(
        #     np.array([click, a, b]))
        if click and reward > 0:
            correct_click += 1
        if reward > 0 and not click:
            missed += 1
        if reward > 0:
            finded += 1
        if terminated or truncated or clicked > 20:
            break
    score = env.SCORE
    print('setp:', step, 'click:', clicked, 'score',
          score, 'correct rate:', correct_click/clicked, 'miss rate:', missed/finded, 'found:', finded)
    return score / env.upperbound
