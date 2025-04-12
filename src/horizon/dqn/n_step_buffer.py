import numpy as np
import torch

from ...config import Config

class NStepBuffer:
    def __init__(self, n_steps: int, device) -> None:
        self.n_steps = n_steps
        self.current_size = 0
        self.position = 0
        self.device = device

        self.states = torch.zeros((self.n_steps, Config.Arch.INPUT_SIZE), dtype=torch.float, device=self.device)
        self.actions = torch.zeros(self.n_steps, dtype=torch.int64, device=self.device)
        self.rewards = torch.zeros(self.n_steps, dtype=torch.float, device=self.device)
        self.gammas = torch.tensor(Config.DQN.GAMMA ** np.arange(self.n_steps), dtype=torch.float, device=self.device)

    def __len__(self):
        return self.current_size

    def clear(self):
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()

        self.current_size = 0
        self.position = 0

    def get_transition(self):
        idx = (self.position - self.current_size) % self.n_steps
        return self.states[idx], self.actions[idx], self.cumulative_reward()

    def pop_transition(self):
        if self.current_size > 0:
            self.current_size -= 1

    def is_full(self):
        return self.current_size == self.n_steps

    def is_empty(self):
        return self.current_size == 0

    def cumulative_reward(self):
        indices = [(self.position - self.current_size + i) % self.n_steps for i in range(self.current_size)]

        rewards_tensor = self.rewards[indices].to(self.device)
        gammas_tensor = self.gammas[:self.current_size].to(self.device)
        return torch.sum(rewards_tensor * gammas_tensor)

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward

        self.position = (self.position + 1) % self.n_steps
        if self.current_size < self.n_steps:
            self.current_size += 1
