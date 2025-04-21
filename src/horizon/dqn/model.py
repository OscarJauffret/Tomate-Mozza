import torch
import torch.nn as nn
import copy
from .noisy_linear import NoisyLinear
from ...config import Config

class Model(nn.Module):
    def __init__(self, device, n_quantiles, cosine_embedding_dim):
        # IQN implementation
        self.device = device
        self.n_quantiles = n_quantiles
        self.cosine_embedding_dim = cosine_embedding_dim

        super(Model, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(Config.Arch.INPUT_SIZE, Config.Arch.LAYER_SIZES[0]),
            nn.ReLU(),
            nn.Linear(Config.Arch.LAYER_SIZES[0], Config.Arch.LAYER_SIZES[1]),
            nn.ReLU(),
        )

        self.phi = nn.Sequential(
            nn.Linear(self.cosine_embedding_dim, Config.Arch.LAYER_SIZES[1]),
            nn.ReLU(),
        )

        self.final = NoisyLinear(Config.Arch.LAYER_SIZES[1], Config.Arch.OUTPUT_SIZE)

    def forward(self, state, taus=None):
        """
        Forward pass of the model
        :param state: The input tensor. It should be of shape (batch_size, Config.Arch.INPUT_SIZE)
        :param taus: The quantile values. It should be of shape (batch_size, n_quantiles). If None, it will be generated.
        :return: The output tensor
        """
        if taus is None:
            taus = self.generate_taus(state.shape[0], uniform=True)
            # Taus are uniformly distributed between 0 and 1 ([[1/n_quantiles, 2/n_quantiles, ..., (n_quantiles-1)/n_quantiles], same, same, ...]) for each batch
            # Shape: (batch_size, n_quantiles)

        assert state.shape[0] == taus.shape[0], "Use same batch sizes for state and tau tensors."
        batch_size, n_quantiles = taus.shape
        assert n_quantiles == self.n_quantiles, f"Expected {self.n_quantiles} quantiles, but got {n_quantiles}."

        state = self.feature(state) # Shape: (batch_size, hidden)
        state = state.unsqueeze(1)  # Shape: (batch_size, 1, hidden)
        # The idea is that each feature vector will be replicated n_quantiles times. Each copy will be multiplied by a different tau value.
        state = state.expand(-1, n_quantiles, -1)   # Shape: (batch_size, n_quantiles, hidden)

        i_pi = torch.arange(1, self.cosine_embedding_dim + 1, device=self.device).float() * torch.pi #  Shape: (cosine_embedding_dim,). This is just [pi, 2*pi, 3*pi, ..., cosine_embedding_dim*pi]
        cos_embedding = torch.cos(i_pi * taus.unsqueeze(-1))  # Shape: (batch_size, n_quantiles, cosine_embedding_dim). This is just the cosine of the tau values multiplied by pi
        cos_embedding = self.phi(cos_embedding)  # Shape: (batch_size, n_quantiles, hidden)

        combined = state * cos_embedding
        q = self.final(combined)  # Shape: (batch_size, n_quantiles, output_size)

        return q

    def generate_taus(self, batch_size=1, uniform=False):
        if uniform:
            return torch.linspace(0, 1, self.n_quantiles + 2, device=self.device)[1:-1].expand((batch_size, self.n_quantiles))
        return torch.rand((batch_size, self.n_quantiles), device=self.device)

    def reset_noise(self):
        if hasattr(self.final, "reset_noise"):
            self.final.reset_noise()

class Trainer:
    def __init__(self, model, device, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.main_model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        self.criterion = nn.SmoothL1Loss(reduction="none")  # Huber loss

        self.n_quantiles = model.n_quantiles
        self.n_target_quantiles = model.n_quantiles
        self.kappa = Config.DQN.KAPPA

        self.target_model = copy.deepcopy(self.main_model).to(self.device)
        self.target_model.eval()

    def train_step(self, state, action, reward, next_state, done, weights=None):
        """
        Train the model using the given state, action, reward, next state and done flag
        :param state: The state of the environment: Shape: (batch_size, Config.Arch.INPUT_SIZE)
        :param action: The actions taken: Shape: (batch_size)
        :param reward: The rewards received: Shape: (batch_size)
        :param next_state: The next state of the environment: Shape: (batch_size, Config.Arch.INPUT_SIZE)
        :param done: The done flag: Shape: (batch_size)
        :param weights: The weights for the loss function: Shape: (batch_size)
        :return: The td_error
        """

        batch_size = state.size(0)
        taus = self.main_model.generate_taus(batch_size)            # Shape: (batch_size, n_quantiles)
        taus_target = self.target_model.generate_taus(batch_size)   # Shape: (batch_size, n_quantiles)

        all_quantiles = self.main_model(state, taus)                # Shape: (batch_size, n_quantiles, n_actions)
        # We just want the quantiles corresponding to the actions taken, so we gather them
        pred = all_quantiles.gather(2, action.unsqueeze(1).unsqueeze(2).expand(-1, self.n_quantiles, 1))  # (batch_size, n_quantiles, 1)

        with torch.no_grad():
            next_q = self.main_model(next_state, taus_target).mean(dim=1)     # Shape: (batch_size, n_actions)
            best_action = torch.argmax(next_q, dim=1, keepdim=True)     # Shape: (batch_size, 1)

            target_quantiles = self.target_model(next_state, taus_target)     # Shape: (batch_size, n_quantiles, n_actions)
            next_q_values = target_quantiles.gather(2, best_action.unsqueeze(1).expand(-1, self.n_target_quantiles, 1)).transpose(1, 2) # (batch_size, 1, n_target_quantiles)

            target = reward.unsqueeze(1).unsqueeze(2) + (1 - done.unsqueeze(1).unsqueeze(2)) * (self.gamma ** Config.DQN.N_STEPS) * next_q_values  # Shape: (batch_size, 1, n_quantiles)
            target = target.detach()

        # Compute pairwise TD error between pred and target quantiles
        td_error = target - pred  # Shape: (batch_size, n_quantiles, target_n_quantiles)
        huber_loss = self.huber_loss(td_error)  # Shape: (batch_size, n_quantiles, target_n_quantiles)
        quantile_loss = (torch.abs(taus.unsqueeze(2) - (td_error.detach() < 0).float()) * huber_loss)  # (batch_size, n_quantiles, target_n_quantiles)

        per_sample_loss = quantile_loss.mean(dim=(1, 2))
        if weights is not None:
            per_sample_loss *= weights

        loss = per_sample_loss.mean()  # Mean loss over the batch

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        td_errors_mean = td_error.abs().mean(dim=(1, 2))  # one TD error per sample for PER
        return td_errors_mean

    def huber_loss(self, td_error):
        return torch.where(td_error.abs() <= self.kappa, 0.5 * td_error.pow(2), self.kappa * (td_error.abs() - 0.5 * self.kappa))



    def update_target(self):
        # Soft update of target model
        for target_param, main_param in zip(self.target_model.parameters(), self.main_model.parameters()):
            target_param.data.copy_(Config.DQN.TAU * main_param.data + (1 - Config.DQN.TAU) * target_param.data)
