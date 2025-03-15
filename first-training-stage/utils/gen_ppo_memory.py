from collections import deque, namedtuple
import random

Transition = namedtuple('Transition',
                        ('attention_map', 'action', 'action_logprob', 'next_attention_map', 'reward', 'done', 'state_value', 'values'))


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        sam = []
        ids = random.sample(range(len(self.memory)), batch_size)
        for id in ids:
            i = 0
            sam.append(self.memory[id + i])
            state_ = self.memory[id + i].next_attention_map
            while not state_ is None:
                sam.append(self.memory[id + i])
                i += 1
                state_ = self.memory[id + i].next_attention_map
            if len(sam) > (batch_size-1):
                break

        return sam
        # return random.sample(self.memory, batch_size)

    def clean(self):
        self.memory = deque([], maxlen=self.capacity)
