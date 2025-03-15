import os
import random
import time
import gym
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import visual_foraging_gym
from utils.models.clickModel import Actor, TaskEmbedding
from torch.distributions import Categorical
from torchvision.models import VGG16_Weights
from utils.positionalencoding2d import positionalencoding2d
from utils.get_attentionmap import get_attention_map
from utils.generate_mmcovs import generate_mmconvs


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


def test(path, env, vgg_model, scale=20):
    device = torch.device('cuda')
    actor = Actor(12, 4)
    task_embedding = TaskEmbedding()
    # actor = nn.DataParallel(actor)

    CHECK_POINT_PATH = (
        path
    )
    checkpoint = torch.load(CHECK_POINT_PATH)
    actor.load_state_dict(checkpoint["model_state_dict"])
    task_embedding.load_state_dict(checkpoint["embedding_model_state_dict"])
    actor.to(device=device)
    task_embedding.to(device)
    actor.eval()
    # load env config
    with open("visual_foraging_gym/envs/env_config.yml", "r") as file:
        env_config = yaml.safe_load(file)
    size = env_config["variable"]["size"]
    # vgg_model = torch.hub.load(
    #     "pytorch/vision:v0.10.0", "vgg16", weights=VGG16_Weights.DEFAULT
    # )
    model_stimuli = vgg_model.features
    model_stimuli = model_stimuli.to(device)

    pe = positionalencoding2d(512, size, size)
    pe = pe.to(device)
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    observation, info = env.reset()
    filenames = info['filename'][0:4]
    MMconvs, mean, std = generate_mmconvs(filenames, vgg_model)
    # obs_img = transforms.ToPILImage()(observation)
    # obs_img.save("obs_img.jpg")
    fixations = [[8, 8]]
    attention_map = get_attention_map(
        observation, fixations, model_stimuli, MMconvs, env_config, mean, std, pe
    )
    values = torch.tensor(info["value"]).float().unsqueeze(0).to(device)
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

    start = time.perf_counter()
    # scale = 20
    order_idx = [0, 1, 2, 3]
    for i in range(scale * 1):
        # draw
        a = "*" * int(i/scale)
        b = "." * int(100 - i/scale)
        c = i / scale
        dur = time.perf_counter() - start
        # print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c,a,b,dur),end = "")
        # time.sleep(0.1)

        # values = torch.tensor(info["value"]).float().unsqueeze(0).to(device)
        action, _, _, _ = select_action(
            actor, task_embedding, attention_map, values)
        b, a = divmod(int(action.cpu()), size)
        saccades.append(
            [a-fixations[-1][0], b-fixations[-1][1]])
        fixations.append([a, b])
        observation, reward, terminated, truncated, info = env.step(
            np.array([a, b]))
        # random.shuffle(order_idx)
        MMconvs_new = []
        next_values = []
        for o in range(4):
            MMconvs_new.append(MMconvs[order_idx[o]])
            next_values.append(info['value'][order_idx[o]])
        attention_map = get_attention_map(
            observation, fixations, model_stimuli, MMconvs_new, env_config, mean, std, pe
        )
        values = torch.tensor(next_values).float().unsqueeze(0).to(device)
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
            # random.shuffle(order_idx)
            MMconvs_new = []
            next_values = []
            for o in range(4):
                MMconvs_new.append(MMconvs[order_idx[o]])
                next_values.append(info['value'][order_idx[o]])
            attention_map = get_attention_map(
                observation, fixations, model_stimuli, MMconvs_new, env_config, mean, std, pe
            )
            values = torch.tensor(
                next_values, device=device).float().unsqueeze(0)
            attention_map = get_attention_map(
                observation, fixations, model_stimuli, MMconvs_new, env_config, mean, std, pe
            )
    env.close()
    # print('score', score / episode)
    return saccades, scores, click_count, click_observer / episode


if __name__ == '__main__':
    import json

    path = "data/model/unified-fixed-image-no-ecc-control.pt"
    env = gym.make("visual_foraging_gym/VisualForaging-v1.6",
                   render_mode="rgb_array", task_mode=1, values=[2, 4, 8, 16])
    saccades, scores, click_count, click_observer = test(path, env)
    data = {'saccades': saccades, 'scores': scores,
            'click count': click_count, 'click observer': click_observer.tolist()}
    with open('data/test/task2ppo-1-6.json', 'w') as json_file:
        json.dump(data, json_file)

    # env = gym.make("visual_foraging_gym/VisualForaging-v1.6",
    #                render_mode="rgb_array", task_mode=1, values=[1,2,3,4])
    # saccades, scores, click_count, click_observer = test(path, env)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/test/task2ppo-g1.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # env = gym.make("visual_foraging_gym/VisualForaging-v1.6",
    #                render_mode="rgb_array", task_mode=1, values=[1,2,4,8])
    # saccades, scores, click_count, click_observer = test(path, env)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/test/task2ppo-g2.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # env = gym.make("visual_foraging_gym/VisualForaging-v1.6",
    #                render_mode="rgb_array", task_mode=1, values=[1,3,5,7])
    # saccades, scores, click_count, click_observer = test(path, env)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/test/task2ppo-g3.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # env = gym.make("visual_foraging_gym/VisualForaging-v1.6",
    #                render_mode="rgb_array", task_mode=1, values=[8,9,10,11])
    # saccades, scores, click_count, click_observer = test(path, env)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/test/task2ppo-g4.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # env = gym.make("visual_foraging_gym/VisualForaging-v1.6",
    #                render_mode="rgb_array", task_mode=1, values=[8,9,13,25])
    # saccades, scores, click_count, click_observer = test(path, env)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/test/task2ppo-g5.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # env = gym.make("visual_foraging_gym/VisualForaging-v1.6",
    #                render_mode="rgb_array", task_mode=1, values=[8,16,32,64])
    # saccades, scores, click_count, click_observer = test(path, env)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/test/task2ppo-g6.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # env = gym.make("visual_foraging_gym/VisualForaging-v1.6",
    #                render_mode="rgb_array", task_mode=1, values=[16,18,20,22])
    # saccades, scores, click_count, click_observer = test(path, env)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/test/task2ppo-g7.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # env = gym.make("visual_foraging_gym/VisualForaging-v1.6",
    #                render_mode="rgb_array", task_mode=1, values=[16,32,64])
    # saccades, scores, click_count, click_observer = test(path, env)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/test/task2ppo-g8.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # saccades, scores, click_count, click_observer = test(path, 1)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/test/task2ppoa-100-1.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # saccades, scores, click_count, click_observer = test(path, 2)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/test/task3ppoa-100-1.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # saccades, scores, click_count, click_observer = test(path, 1)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/test/task2ppou.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # saccades, scores, click_count, click_observer = test(path, 2)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/test/task3ppou.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # path = "data/model/task1ppo.pt"
    # saccades, scores, click_count, click_observer = test(path, 0)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/test/task1ppo.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # path = "data/model/task2ppo.pt"
    # saccades, scores, click_count, click_observer = test(path, 1)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/test/task2ppo.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # path = "data/model/task3ppo2.pt"
    # saccades, scores, click_count, click_observer = test(path, 2)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/test/task3ppo2.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # path = "data/ppo/task3_model.pt"
    # saccades, scores, click_count, click_observer = test(path, 0)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/ppo/task31.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # path = "data/ppo/task2_model.pt"
    # saccades, scores, click_count, click_observer = test(path, 1)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/ppo/task2.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # path = "data/ppo/task3_model.pt"
    # saccades, scores, click_count, click_observer = test(path, 1)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/ppo/task32.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # path = "data/ppo/task3_model.pt"
    # saccades, scores, click_count, click_observer = test(path, 2)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/ppo/task3.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # path = "data/ppo/task1augmentation.pt"
    # saccades, scores, click_count, click_observer = test(path, 0)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/ppo/task1withoutaugmentation.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # path = "data/ppo/task2augmentation.pt"
    # saccades, scores, click_count, click_observer = test(path, 1)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/ppo/task2withoutaugmentation.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # path = "data/different hyperparameter/task1augmentation.pt"
    # saccades, scores, click_count, click_observer = test(path, 0)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/different hyperparameter/task1augmentation.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # path = "data/different hyperparameter/task2augmentation3.pt"
    # saccades, scores, click_count, click_observer = test(path, 1)
    # data = {'saccades': saccades, 'scores': scores,
    #         'click count': click_count, 'click observer': click_observer.tolist()}
    # with open('data/different hyperparameter/task2augmentation.json', 'w') as json_file:
    #     json.dump(data, json_file)
