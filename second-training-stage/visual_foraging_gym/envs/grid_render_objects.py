# Import the pygame module
import os

import pygame
from pygame import Vector2

# Import random for random numbers
import random

# Import pygame.locals for easier access to key coordinates
# Updated to conform to flake8 and black standards
from pygame.locals import (
    RLEACCEL,
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)


# Define a player object by extending pygame.sprite.Sprite
# The surface drawn on the screen is now an attribute of 'player'
class Player(pygame.sprite.Sprite):
    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT):
        super(Player, self).__init__()
        self.surf = pygame.Surface((5, 5))
        self.surf.fill((0, 0, 0))
        self.rect = self.surf.get_rect()
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

    # Move the sprite based on user keypresses
    def update(self, mx, my):
        self.rect.move_ip(mx - self.rect[0], my - self.rect[1])

        # Keep player on the screen
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > self.SCREEN_WIDTH:
            self.rect.right = self.SCREEN_WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= self.SCREEN_HEIGHT:
            self.rect.bottom = self.SCREEN_HEIGHT


# Define a target Object, load random picture
class Target(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super(Target, self).__init__()
        dir = 'visual_foraging_gym/envs/target'
        file_list = []
        for img in os.listdir(dir):
            if img.endswith("jpg"):
                file_list.append(img)
        filename = random.choice(file_list)
        self.surf = pygame.image.load(os.path.join(dir, filename)).convert()
        self.surf = pygame.transform.rotozoom(self.surf, 0, 0.25)
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.rect = self.surf.get_rect(
            center=(
                x,
                y,
            )
        )

    def update(self, click_x, click_y):
        if self.rect.collidepoint(click_x, click_y):
            self.kill()


class TargetOne(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super(TargetOne, self).__init__()
        self.surf = pygame.image.load(
            'visual_foraging_gym/envs/target/045.jpg').convert()
        self.surf = pygame.transform.rotozoom(self.surf, 0, 0.25)
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.rect = self.surf.get_rect(
            center=(
                x,
                y,
            )
        )

    def update(self, click_x, click_y):
        if self.rect.collidepoint(click_x, click_y):
            self.kill()


class TargetTwo(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super(TargetTwo, self).__init__()
        self.surf = pygame.image.load(
            'visual_foraging_gym/envs/target/8055181.thl.jpg').convert()
        self.surf = pygame.transform.rotozoom(self.surf, 0, 0.25)
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.rect = self.surf.get_rect(
            center=(
                x,
                y,
            )
        )

    def update(self, click_x, click_y):
        if self.rect.collidepoint(click_x, click_y):
            self.kill()


class TargetThree(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super(TargetThree, self).__init__()
        self.surf = pygame.image.load(
            'visual_foraging_gym/envs/target/8059226.thl.jpg').convert()
        self.surf = pygame.transform.rotozoom(self.surf, 0, 0.25)
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.rect = self.surf.get_rect(
            center=(
                x,
                y,
            )
        )

    def update(self, click_x, click_y):
        if self.rect.collidepoint(click_x, click_y):
            self.kill()


class TargetFour(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super(TargetFour, self).__init__()
        self.surf = pygame.image.load(
            'visual_foraging_gym/envs/target/8077486.thl.jpg').convert()
        self.surf = pygame.transform.rotozoom(self.surf, 0, 0.25)
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.rect = self.surf.get_rect(
            center=(
                x,
                y,
            )
        )

    def update(self, click_x, click_y):
        if self.rect.collidepoint(click_x, click_y):
            self.kill()


# Define a distractor Object, load random picture
class Distractor(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super(Distractor, self).__init__()
        dir = 'visual_foraging_gym/envs/distractor'
        file_list = []
        for img in os.listdir(dir):
            if img.endswith("jpg"):
                file_list.append(img)
        filename = random.choice(file_list)
        self.surf = pygame.image.load(os.path.join(dir, filename)).convert()
        self.surf = pygame.transform.rotozoom(self.surf, 0, 0.25)
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.rect = self.surf.get_rect(
            center=(
                x,
                y,
            )
        )

    def update(self, click_x, click_y):
        if self.rect.collidepoint(click_x, click_y):
            self.kill()


# Define the score object
class Score(pygame.sprite.Sprite):
    def __init__(self, SCORE, font, SCREEN_WIDTH, SCREEN_HEIGHT):
        super(Score, self).__init__()
        self.score = SCORE
        self.font = font
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.surf = font.render('Score: ' + str(self.score), True, (0, 0, 0))
        self.rect = self.surf.get_rect(
            center=(
                (SCREEN_WIDTH) / 2,
                (SCREEN_HEIGHT) / 2,
            )
        )

    def update(self, SCORE, SCORE_COLOR):
        self.score = SCORE
        self.surf = self.font.render(
            'Score: ' + str(self.score), True, SCORE_COLOR)
        self.rect = self.surf.get_rect(
            center=(
                (self.SCREEN_WIDTH - self.surf.get_width()) / 2,
                (self.SCREEN_HEIGHT - self.surf.get_height()) / 2,
            )
        )
        self.rect = self.surf.get_rect(
            center=(
                (self.SCREEN_WIDTH - self.surf.get_width()) / 2,
                (self.SCREEN_HEIGHT - self.surf.get_height()) / 2,
            )
        )


# Define Patch button object
class Patch(pygame.sprite.Sprite):
    def __init__(self, font, SCREEN_WIDTH, SCREEN_HEIGHT):
        super(Patch, self).__init__()
        self.surf = font.render("next patch", True, (0, 0, 0))
        self.rect = self.surf.get_rect(
            center=(
                (SCREEN_WIDTH - self.surf.get_width()) / 2,
                (SCREEN_HEIGHT - self.surf.get_height()) /
                2 + self.surf.get_height(),
            )
        )
