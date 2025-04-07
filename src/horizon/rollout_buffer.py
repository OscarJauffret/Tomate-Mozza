import numpy as np
import torch

from ..config import Config

class RolloutBuffer:
    def __init__(self, device):
        self.device = device

        self.states = torch.zeros((Config.NN.MEMORY_SIZE, Config.NN.Arch.INPUT_SIZE), dtype=torch.float, device=self.device)
        self.actions = torch.zeros(Config.NN.MEMORY_SIZE, dtype=torch.int64, device=self.device)
        self.probs = torch.zeros(Config.NN.MEMORY_SIZE, dtype=torch.float, device=self.device)
        self.rewards = torch.zeros(Config.NN.MEMORY_SIZE, dtype=torch.float, device=self.device)
        self.dones = torch.zeros(Config.NN.MEMORY_SIZE, dtype=torch.int, device=self.device)
        self.values = torch.zeros(Config.NN.MEMORY_SIZE, dtype=torch.float, device=self.device)

        self.batch_size = Config.NN.BATCH_SIZE
        self.position = 0

    def is_full(self):
        return self.position >= Config.NN.MEMORY_SIZE

    def clear(self):
        self.states.zero_()
        self.actions.zero_()
        self.probs.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.values.zero_()

        self.position = 0

    def add(self, state, action, log_prob, reward, done, value):
        assert self.position < Config.NN.MEMORY_SIZE, "Rollout buffer is full. Please clear it before adding new data."

        self.states[self.position] = state.detach()
        self.actions[self.position] = action.detach()
        self.probs[self.position] = log_prob.detach()
        self.rewards[self.position] = reward.detach()
        self.dones[self.position] = done.detach()
        self.values[self.position] = value.detach()
        self.position += 1

    def generate_batches(self):
        assert self.is_full(), "Rollout buffer is not full. Please fill it before generating batches."

        batch_start = np.arange(0, self.position, self.batch_size)
        indices = np.arange(self.position, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start] # batches is a list of arrays of indices. These indices indicate a batch of states, actions, probs, values, rewards and dones
        return self.states, self.actions, self.probs, self.values, self.rewards, self.dones, batches
