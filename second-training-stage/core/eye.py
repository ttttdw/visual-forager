import random

import torch
from utils.generate_mmcovs import generate_mmconvs
from utils.get_attentionmap import get_attention_map, get_eccattention_map, get_perfect_map


class Eye:
    def __init__(self, vgg_model, visual_model, device, env_config, if_shuffle=False, ecc_mode=False, forgetting=0.8) -> None:
        self.device = device
        self.vgg_model = vgg_model.to(device)
        self.fixations = [[8, 8]]
        self.visual_model = visual_model.to(device)
        self.if_shuffle = if_shuffle
        self.ecc_mode = ecc_mode
        self.env_config = env_config
        self.forgetting = forgetting

    def reset(self, info):
        """
        Reset working memory at the begining of each episodes
        """
        self.points = info['value']
        filenames = info['filename'][0:4]
        self.MMconvs, self.mean, self.std = generate_mmconvs(
            filenames, self.vgg_model)
        self.fixations = [[8, 8]]

    def visual_process(self, observation):
        """
        Input observation, output similarity maps
        """
        order_idx = [0, 1, 2, 3]
        if self.if_shuffle:
            random.shuffle(order_idx)
        MMconvs_new = []
        next_points = []
        for o in range(4):
            MMconvs_new.append(self.MMconvs[order_idx[o]])
            next_points.append(self.points[order_idx[o]])
        if self.ecc_mode:
            similarity_map = get_eccattention_map(
                observation, self.fixations, self.visual_model, MMconvs_new, self.env_config, self.mean, self.std, None, forgetting=self.forgetting
            )
        else:
            similarity_map = get_attention_map(
                observation, self.fixations, self.visual_model, MMconvs_new, self.env_config, self.mean, self.std, None, forgetting=self.forgetting
            )
        points = torch.tensor(
            next_points, device=self.device).float().unsqueeze(0)

        return similarity_map, points

class PerfectEye(Eye):
    def visual_process(self, observation):
        """
        Input observation, output similarity maps
        """
        order_idx = [0, 1, 2, 3]
        if self.if_shuffle:
            random.shuffle(order_idx)
        observation_new = torch.zeros((4, 16, 16), device=self.device)
        next_points = []
        for o in range(4):
            observation_new[o,:,:] = observation[order_idx[o],:,:]
            next_points.append(self.points[order_idx[o]])
        similarity_map = get_perfect_map(
                observation_new, self.fixations, self.env_config, forgetting=self.forgetting
            )
        points = torch.tensor(
            next_points, device=self.device).float().unsqueeze(0)

        return similarity_map, points