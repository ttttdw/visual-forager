import os
import random

import gym
import torch
import yaml
from gym import spaces
import pygame
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
from visual_foraging_gym.envs.grid_render_objects import Target, TargetOne, TargetTwo, TargetThree, TargetFour, Distractor, Player, Patch, Score

transform = transforms.Compose([transforms.PILToTensor()])


class GridVisualForagingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, task_mode=1):

        self.SCORE = 0

        self.rewards = [[4, 4, 4, 4], [2, 4, 8, 16], [2, 4, 8, 16]]
        self.popularities = [[16, 8, 4, 2], [8, 8, 8, 8], [16, 8, 4, 2]]
        self.task_mod = task_mode  # random.choice([0, 1, 2])

        self.target_set = []
        self.distractor_set = []
        self.device = torch.device('cuda')

        # Stimuli file list
        folder_path = 'visual_foraging_gym/envs/TargetFixed'
        all_files = os.listdir(folder_path)
        self.target_image_file_list = [
            file for file in all_files if file.lower().endswith(('.jpg'))]
        folder_path = 'visual_foraging_gym/envs/DistractorFixed'
        all_files = os.listdir(folder_path)
        self.distractor_image_file_list = [
            file for file in all_files if file.lower().endswith(('.jpg'))]

        self.saccade_map = Image.open(
            "visual_foraging_gym/envs/naturaldesign_2Dsaccadeprior.jpg")
        self.saccade_map = transform(self.saccade_map) / 255

        # load config
        with open('visual_foraging_gym/envs/large_env_config.yml', 'r') as file:
            config = yaml.safe_load(file)
        self.size = config['variable']['size']
        self.pix_size = config['variable']['target size']
        self.SCREEN_WIDTH = self.size * self.pix_size
        self.SCREEN_HEIGHT = self.size * self.pix_size

        self.n_distractor = 75
        self.n_item = config['variable']['total number']

        # click history
        self.clicks = {'click_target_one': 0,
                       'click_target_two': 0,
                       'click_target_three': 0,
                       'click_target_four': 0,
                       'click_distractor': 0
                       }
        self.now_click = None
        # initial fixation
        self.fixations = [
            [(round((self.size - 1) / 2) + 0.5) * self.pix_size, (round((self.size - 1) / 2) + 0.5) * self.pix_size]]

        pygame.init()
        self.font = pygame.font.SysFont("Arial", 14)
        # Instantiate player. Right now, this is just a rectangle.
        self.player = Player(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        # self.patch = Patch(self.font, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.score = Score(self.SCORE, self.font,
                           self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.targets = pygame.sprite.Group()
        self.distractors = pygame.sprite.Group()
        self.all_sprites = pygame.sprite.Group()
        self.screen = pygame.display.set_mode(
            (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.action_space = spaces.Discrete(self.size)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
                If human-rendering is used, `self.window` will be a reference
                to the window that we draw to. `self.clock` will be a clock that is used
                to ensure that the environment is rendered at the correct framerate in
                human-mode. They will remain `None` until human-mode is used for the
                first time.
                """
        self.window = None
        self.clock = None

    def _get_obs(self):
        if self.render_mode == "human":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            # Make new canvas
            canvas = torch.ones(
                (self.size*self.pix_size, self.size*self.pix_size, 3), device=self.device)
            canvas = canvas * 128

            # Paste target one
            for position in self.target_ones:
                render_position = position * self.pix_size
                x_ = int(position[1] * self.pix_size)
                x = int((position[1]+1) * self.pix_size)
                y_ = int(position[0] * self.pix_size)
                y = int((position[0]+1) * self.pix_size)
                canvas[x_:x, y_:y, :] = self.target_set[0]

            # Paste target two
            for position in self.target_twos:
                render_position = position * self.pix_size
                x_ = int(position[1] * self.pix_size)
                x = int((position[1]+1) * self.pix_size)
                y_ = int(position[0] * self.pix_size)
                y = int((position[0]+1) * self.pix_size)
                canvas[x_:x, y_:y, :] = self.target_set[1]

            # Paste target three
            for position in self.target_threes:
                render_position = position * self.pix_size
                x_ = int(position[1] * self.pix_size)
                x = int((position[1]+1) * self.pix_size)
                y_ = int(position[0] * self.pix_size)
                y = int((position[0]+1) * self.pix_size)
                canvas[x_:x, y_:y, :] = self.target_set[2]

            # Paste target four
            for position in self.target_fours:
                render_position = position * self.pix_size
                x_ = int(position[1] * self.pix_size)
                x = int((position[1]+1) * self.pix_size)
                y_ = int(position[0] * self.pix_size)
                y = int((position[0]+1) * self.pix_size)
                canvas[x_:x, y_:y, :] = self.target_set[3]

            # Paste distractor
            for i, position in enumerate(self.distractor_positions):
                image = self.distractor_set[i]
                render_position = position * self.pix_size
                x_ = int(position[1] * self.pix_size)
                x = int((position[1]+1) * self.pix_size)
                y_ = int(position[0] * self.pix_size)
                y = int((position[0]+1) * self.pix_size)
                canvas[x_:x, y_:y, :] = image

            # canvas.save("observation_image.jpg")
            return canvas

    def _get_info(self):
        return {'value': self.rewards[self.task_mod], 'click_count': self.clicks, 'now click': self.now_click, 'filename': self.target_image_file_list}

    def _add_target_one(self):
        position = self.all_sprited_positions[0]
        self.all_sprited_positions = [
            arr for arr in self.all_sprited_positions if not np.array_equal(arr, position)]

        self.target_ones.append(position)

        if self.render_mode == 'human':
            render_position = (position + 0.5) * self.pix_size
            new_target = TargetOne(render_position[0], render_position[1])
            self.all_sprites.add(new_target)
            self.targets.add(new_target)

    def _add_target_two(self):
        position = self.all_sprited_positions[0]
        self.all_sprited_positions = [
            arr for arr in self.all_sprited_positions if not np.array_equal(arr, position)]
        self.target_twos.append(position)

        if self.render_mode == 'human':
            render_position = (position + 0.5) * self.pix_size
            new_target = TargetTwo(render_position[0], render_position[1])
            self.all_sprites.add(new_target)
            self.targets.add(new_target)

    def _add_target_three(self):
        position = self.all_sprited_positions[0]
        self.all_sprited_positions = [
            arr for arr in self.all_sprited_positions if not np.array_equal(arr, position)]
        self.target_threes.append(position)

        if self.render_mode == 'human':
            render_position = (position + 0.5) * self.pix_size
            new_target = TargetThree(render_position[0], render_position[1])
            self.all_sprites.add(new_target)
            self.targets.add(new_target)

    def _add_target_four(self):
        position = self.all_sprited_positions[0]
        self.all_sprited_positions = [
            arr for arr in self.all_sprited_positions if not np.array_equal(arr, position)]
        self.target_fours.append(position)

        if self.render_mode == 'human':
            render_position = (position + 0.5) * self.pix_size
            new_target = TargetFour(render_position[0], render_position[1])
            self.all_sprites.add(new_target)
            self.targets.add(new_target)

    def _add_distractor(self):
        position = self.all_sprited_positions[0]
        self.all_sprited_positions = [
            arr for arr in self.all_sprited_positions if not np.array_equal(arr, position)]
        self.distractor_positions.append(position)

        if self.render_mode == 'human':
            render_position = (position + 0.5) * self.pix_size
            new_distractor = Distractor(render_position[0], render_position[1])
            self.all_sprites.add(new_distractor)
            self.distractors.add(new_distractor)

    def _new_patch(self):
        self.all_sprites.add(self.player)
        self.all_sprited_positions = []
        while len(self.all_sprited_positions) < self.n_item:
            position = self.np_random.integers(
                0, self.size, size=2, dtype=int)
            if not any(np.all(position == vector) for vector in self.all_sprited_positions):
                self.all_sprited_positions.append(position)

        self.target_ones = []
        while len(self.target_ones) < self.popularities[self.task_mod][0]:
            self._add_target_one()
        self.target_twos = []
        while len(self.target_twos) < self.popularities[self.task_mod][1]:
            self._add_target_two()
        self.target_threes = []
        while len(self.target_threes) < self.popularities[self.task_mod][2]:
            self._add_target_three()
        self.target_fours = []
        while len(self.target_fours) < self.popularities[self.task_mod][3]:
            self._add_target_four()

        self.distractor_positions = []
        while len(self.distractor_positions) < self.n_distractor:
            self._add_distractor()

    def change_mode(self, task_mode):
        self.task_mod = task_mode

    def change_value(self, values):
        self.rewards[self.task_mod] = values

    def reset(self, seed=None, option=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # random.shuffle(self.rewards[self.task_mod])
        self.rewards[self.task_mod] = [
            random.random()*16, random.random()*16, random.random()*16, random.random()*16]

        # Read stimulis
        folder_path = 'visual_foraging_gym/envs/TargetFixed'
        random.shuffle(self.target_image_file_list)
        self.target_set = []
        for file in self.target_image_file_list:
            image = Image.open(folder_path + '/' + file)
            image = image.resize((self.pix_size, self.pix_size))
            image = np.array(image)
            image = torch.from_numpy(image).to(self.device)
            self.target_set.append(image)

        folder_path = 'visual_foraging_gym/envs/DistractorFixed'
        random.shuffle(self.distractor_image_file_list)
        self.distractor_set = []
        for file in self.distractor_image_file_list:
            image = Image.open(folder_path + '/' + file)
            image = image.resize((self.pix_size, self.pix_size))
            image = np.array(image)
            image = torch.from_numpy(image).to(self.device)
            self.distractor_set.append(image)

        distractor_set = []
        for i in range(self.n_distractor):
            distractor_set.append(random.choice(self.distractor_set))
        self.distractor_set = distractor_set

        self.fixations = [
            [(round((self.size - 1) / 2) + 0.5) * self.pix_size, (round((self.size - 1) / 2) + 0.5) * self.pix_size]]
        # click history
        self.clicks = {'click_target_one': 0,
                       'click_target_two': 0,
                       'click_target_three': 0,
                       'click_target_four': 0,
                       'click_distractor': 0
                       }
        self.now_click = None
        self.SCORE = 0
        if self.render_mode == 'human':
            self.SCORE_COLOR = (0, 0, 0)
            self.all_sprites.empty()
            self.targets.empty()
            self.distractors.empty()
            self.player.update(0, 0)
        self._new_patch()

        if self.render_mode == "human":
            self._render_frame()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def step(self, action):

        click_point = (action + 0.5) * self.pix_size
        click_point_x = click_point[0]
        click_point_y = click_point[1]
        saccade_x = int(self.fixations[-1][0] - click_point_x) + 1024
        saccade_y = int(self.fixations[-1][1] - click_point_y) + 1024
        self.fixations.append([click_point_x, click_point_y])

        if any(np.all(action == vector) for vector in self.target_ones):
            reward = self.rewards[self.task_mod][0]
            self.SCORE += self.rewards[self.task_mod][0]
            self.SCORE_COLOR = (0, 0, 0)
            self.target_ones = [
                arr for arr in self.target_ones if not np.array_equal(arr, action)
            ]
            self.clicks['click_target_one'] += 1
            self.now_click = 0
        elif any(np.all(action == vector) for vector in self.target_twos):
            reward = self.rewards[self.task_mod][1]
            self.SCORE += self.rewards[self.task_mod][1]
            self.SCORE_COLOR = (0, 0, 0)
            self.target_twos = [
                arr for arr in self.target_twos if not np.array_equal(arr, action)
            ]
            self.clicks['click_target_two'] += 1
            self.now_click = 1
        elif any(np.all(action == vector) for vector in self.target_threes):
            reward = self.rewards[self.task_mod][2]
            self.SCORE += self.rewards[self.task_mod][2]
            self.SCORE_COLOR = (0, 0, 0)
            self.target_threes = [
                arr for arr in self.target_threes if not np.array_equal(arr, action)
            ]
            self.clicks['click_target_three'] += 1
            self.now_click = 2
        elif any(np.all(action == vector) for vector in self.target_fours):
            reward = self.rewards[self.task_mod][3]
            self.SCORE += self.rewards[self.task_mod][3]
            self.SCORE_COLOR = (0, 0, 0)
            self.target_fours = [
                arr for arr in self.target_fours if not np.array_equal(arr, action)
            ]
            self.clicks['click_target_four'] += 1
            self.now_click = 3
        elif any(np.all(action == vector) for vector in self.distractor_positions):
            reward = -1
            self.SCORE += -1
            self.SCORE_COLOR = (255, 0, 0)

            self.clicks['click_distractor'] += 1
            self.now_click = 4
        else:
            reward = -0.1

        if self.render_mode == "human":
            # Update the player sprite based on user keypresses
            self.player.update(click_point_x, click_point_y)
            self.targets.update(click_point_x, click_point_y)
            self.score.update(self.SCORE, self.SCORE_COLOR)
            # terminated = False if self.SCORE < self.ENOUGH_SCORE else True
            terminated = len(self.targets) == 0
            self._render_frame()

        observation = self._get_obs()
        # random.shuffle(self.rewards[self.task_mod])
        info = self._get_info()

        return observation, reward, False, False, info

    def _render_frame(self):
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # The following line copies our drawings from `canvas` to the visible window
        # Fill the screen with white
        self.screen.fill((255, 255, 255))

        # Draw all sprites
        for entity in self.all_sprites:
            self.screen.blit(entity.surf, entity.rect)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        pygame.display.quit()
        pygame.quit()
