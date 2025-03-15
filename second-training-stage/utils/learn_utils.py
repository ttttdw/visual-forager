import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def select_action(actor, attention_map, values):
    policy, click_p, value = actor(attention_map, values)
    policy = nn.functional.softmax(policy, 1)
    policy = policy.squeeze()
    dist = Categorical(policy)
    c_dist = Categorical(torch.cat((click_p, 1-click_p)).squeeze())

    action = dist.sample()
    click = c_dist.sample()
    action_logprob = dist.log_prob(action) + c_dist.log_prob(click)
    action_entropy = dist.entropy()+c_dist.entropy()

    return [click.item(), action.item()], action_logprob, value.squeeze(), action_entropy


def another_select_action(actor, task_embedding, attention_map, values):
    # split task embedding and ViT
    values = task_embedding(values)
    policy, click_p, value = actor(attention_map, values)
    policy = nn.functional.softmax(policy, 1)
    policy = policy.squeeze()
    dist = Categorical(policy)
    c_dist = Categorical(torch.cat((click_p, 1-click_p)).squeeze())

    action = dist.sample()
    click = c_dist.sample()
    action_logprob = dist.log_prob(action) + c_dist.log_prob(click)
    action_entropy = dist.entropy()+c_dist.entropy()

    return [click.item(), action.item()], action_logprob, value.squeeze(), action_entropy


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
