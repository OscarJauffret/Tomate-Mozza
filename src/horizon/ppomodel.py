import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

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
        advantages = []
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[step + 1] * (1 - dones[step])
            delta = rewards[step] + self.gamma * next_value - values[step]
            advantages.insert(0, delta + self.gamma * self.gae_lambda * advantages[0] if advantages else delta)
        advantages = np.array(advantages)
        return torch.from_numpy(advantages).to(self.device)  # Convert to tensor and move to device

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
        states, actions, probs, values, rewards, dones, batches = memory.generate_batches()
        # Convert to tensors and move to device
        values = torch.from_numpy(values).to(self.device)  # Shape: [memory_size, 1]
        values = values.squeeze(1)  # Shape: [memory_size]
        advantages = self.compute_gae(rewards, values, dones)   # Shape [memory_size]

        for batch in batches:
            batch_states = torch.tensor(states[batch], dtype=torch.float, device=self.device)   # Now we randomize the order of the transitions
            batch_old_probs = torch.tensor(probs[batch], dtype=torch.float, device=self.device)
            batch_actions = torch.tensor(actions[batch], dtype=torch.float, device=self.device)
            # batch_states is a tensor of shape [batch_size, state_size]
            # batch_old_probs is a tensor of shape [batch_size]
            # batch_actions is a tensor of shape [batch_size]

            dist = self.actor(batch_states) # Shape: [batch_size, n_actions]
            critic_value = self.critic(batch_states).squeeze(1)# Shap: [batch_size]

            new_probs = dist.log_prob(batch_actions)    # Shape: [batch_size]
            prob_ratio = torch.exp(new_probs - batch_old_probs)
            weighted_probs = prob_ratio * advantages[batch] # Shape: [batch_size]
            clipped_probs = torch.clamp(prob_ratio, 1 - Config.NN.EPSILON, 1 + Config.NN.EPSILON) * advantages[batch]   # Shape: [batch_size]

            actor_loss = -torch.min(weighted_probs, clipped_probs).mean()   # Shape: [1]
            returns = self.compute_returns(advantages[batch], values[batch])

            critic_loss = self.mse(critic_value, returns)

            total_loss = actor_loss + Config.NN.C1 * critic_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
