from collections import deque, namedtuple
import random
import torch
import copy

Transition = namedtuple('Transition',
                        ('attention_map', 'action', 'action_logprob', 'next_attention_map', 'reward', 'done', 'state_value', 'values'))


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.advantages = None

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):

        index = random.sample(range(len(self.memory)), batch_size)
        sam = []
        advantages = []

        for i in index:
            sam.append(self.memory[i])
            advantages.append(self.advantages[i])

        return sam, advantages

    def clean(self):
        self.memory = deque([], maxlen=self.capacity)
        self.advantages = None

    def advantage_estimation(self, gamma=0.99, lam=0.95):
        T = len(self.memory)
        advantages = torch.zeros((T, 1))
        advantage = 0
        next_value = 0

        for t in reversed(range(T)):
            done = self.memory[t].done
            td_error = self.memory[t].reward + next_value * \
                (1 - done) * gamma - self.memory[t].state_value
            advantage = td_error + advantage * gamma * lam * (1 - done)
            next_value = copy.deepcopy(self.memory[t].state_value)
            advantages[t] = advantage

        self.advantages = (advantages - advantages.mean()) / advantages.std()
