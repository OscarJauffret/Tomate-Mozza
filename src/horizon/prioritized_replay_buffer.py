import numpy as np
from ..config import Config

class PrioritizedReplayBuffer:

    def __init__(self, capacity, alpha, beta):
        self.capacity = capacity
        self.buffer = np.zeros(self.capacity, dtype=object)
        self.priorities = np.zeros(self.capacity)
        self.pos = 0
        self.fill_level = 0

        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6

    def __len__(self):
        return self.fill_level

    def add(self, transition, priority=None):
        if priority is None:
            priority = max(self.priorities, default=1.0)

        self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        self.fill_level = min(self.fill_level + 1, self.capacity)

    def sample(self, batch_size):
        if self.fill_level < Config.NN.MIN_MEMORY:
            return None

        self.beta += (Config.NN.BETA_MAX - Config.NN.BETA_START) / Config.NN.BETA_INCREMENT_STEPS

        valid_priorities = self.priorities[:self.fill_level]
        scaled = (valid_priorities + self.epsilon) ** self.alpha
        probabilities = scaled / scaled.sum()

        indices = np.random.choice(self.fill_level, batch_size, p=probabilities)
        samples = self.buffer[indices]

        weights = ((1 / self.fill_level) * (1 / probabilities[indices])) ** self.beta
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


