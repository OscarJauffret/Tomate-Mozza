import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from itertools import repeat

from .rollout_buffer import RolloutBuffer
from ..config import Config

class PPOActor(nn.Module):
    def __init__(self):
        super(PPOActor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(Config.NN.Arch.INPUT_SIZE, Config.NN.Arch.LAYER_SIZES[0]),
            nn.ReLU(),
            nn.Linear(Config.NN.Arch.LAYER_SIZES[0], Config.NN.Arch.LAYER_SIZES[1]),
            nn.ReLU(),
            nn.Linear(Config.NN.Arch.LAYER_SIZES[1], Config.NN.Arch.OUTPUT_SIZE),
            nn.Softmax(dim=-1)  # Softmax for action probabilities
        )

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)        # Creates a categorical distribution from the action probabilities (discrete)
        return dist

    def save_checkpoint(self, file):
        torch.save(self.state_dict(), file)

    def load_checkpoint(self, file):
        self.load_state_dict(torch.load(file))


class PPOCritic(nn.Module):
    def __init__(self):
        super(PPOCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(Config.NN.Arch.INPUT_SIZE, Config.NN.Arch.LAYER_SIZES[0]),
            nn.ReLU(),
            nn.Linear(Config.NN.Arch.LAYER_SIZES[0], Config.NN.Arch.LAYER_SIZES[1]),
            nn.ReLU(),
            nn.Linear(Config.NN.Arch.LAYER_SIZES[1], 1)  # Output a single value (state value)
        )

    def forward(self, state):
        return self.critic(state)

    def save_checkpoint(self, file):
        torch.save(self.state_dict(), file)

    def load_checkpoint(self, file):
        self.load_state_dict(torch.load(file))


class PPOTrainer:
    def __init__(self, actor: PPOActor, critic: PPOCritic, device: torch.device):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=Config.NN.LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=Config.NN.LEARNING_RATE)
        self.device = device
        self.mse = nn.MSELoss()

        self.gamma = Config.NN.GAMMA
        self.gae_lambda = Config.NN.LAMBDA

    def compute_gae(self, rewards, values, dones):
        """
        Compute the GAE. It is used directly to compute the loss for the actor, and the critic sums it with the values to compute its loss
        :param rewards: the rewards received
        :param values: the values predicted by the critic
        :param dones: the done flags
        :return: the GAE
        """
        advantages = torch.zeros_like(rewards, dtype=torch.float, device=self.device)  # Shape: [memory_size]
        for step in reversed(range(len(rewards))):
            if dones[step] or step == len(rewards) - 1:
                delta = rewards[step] - values[step]
                advantages[step] = delta
            else:
                next_value = values[step + 1] * (1 - dones[step])
                delta = rewards[step] + self.gamma * next_value - values[step]
                advantages[step] = delta + self.gamma * self.gae_lambda * advantages[step + 1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize the advantages
        return advantages  # Shape: [memory_size]

    def compute_returns(self, advantages, values):
        """
        The returns are used to compute the loss for the critic. It is the sum of the advantages and the values predicted by the critic
        :param advantages: the GAE
        :param values: the values predicted by the critic
        :return: the returns
        """
        return advantages + values

    def train_step(self, memory: RolloutBuffer):
        """
        Train the actor and the critic using the data in the memory
        :param memory: the memory containing the data
        """
        states, actions, probs, values, rewards, dones = memory.get_buffer()
        advantages = self.compute_gae(rewards, values, dones)   # Shape [memory_size]
        returns = self.compute_returns(advantages, values)
        for _ in repeat(None, Config.NN.EPOCHS):
            batches = memory.generate_batches()
            for batch in batches:
                batch_states = states[batch]  # Shape: [batch_size, state_size]
                batch_old_probs = probs[batch]  # Shape: [batch_size]
                batch_actions = actions[batch] # Shape: [batch_size]

                dist = self.actor(batch_states) # Shape: [batch_size, n_actions]
                critic_value = self.critic(batch_states).squeeze(1) # Shape: [batch_size]

                new_probs = dist.log_prob(batch_actions)    # Shape: [batch_size]
                prob_ratio = torch.exp(new_probs - batch_old_probs)
                weighted_probs = prob_ratio * advantages[batch] # Shape: [batch_size]
                clipped_probs = torch.clamp(prob_ratio, 1 - Config.NN.EPSILON, 1 + Config.NN.EPSILON) * advantages[batch]   # Shape: [batch_size]
                actor_loss = (-torch.min(weighted_probs, clipped_probs)).mean()   # Shape: [1]

                critic_loss = self.mse(critic_value, returns[batch])

                entropy = dist.entropy().mean()  # Shape: [1]

                total_loss = actor_loss + Config.NN.C1 * critic_loss - Config.NN.C2 * entropy
                # In the original paper, the objective is to maximize the expected return (actor_loss - Config.NN.C1 * critic_loss + Config.NN.C2 * entropy).
                # However, in PyTorch, we minimize the loss.
                # This means that we need to minimize the negative of the expected return.(-actor_loss + Config.NN.C1 * critic_loss - Config.NN.C2 * entropy)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
