import torch
import torch.nn as nn
from ..config import Config

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(Config.NN.Arch.INPUT_SIZE, Config.NN.Arch.LAYER_SIZES[0])
        self.layer2 = nn.Linear(Config.NN.Arch.LAYER_SIZES[0], Config.NN.Arch.LAYER_SIZES[1])
        self.layer3 = nn.Linear(Config.NN.Arch.LAYER_SIZES[1], Config.NN.Arch.OUTPUT_SIZE)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

class QTrainer:
    def __init__(self, model, device, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = state.to(self.device) if state.device != self.device else state
        next_state = next_state.to(self.device) if next_state.device != self.device else next_state
        action = action.to(self.device) if action.device != self.device else action
        reward = reward.to(self.device) if reward.device != self.device else reward

        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # Predicted Q values
        pred = self.model(state)
        target = pred.clone()

        max_next_q_values = self.model(next_state).max(dim=1).values

        # If done, the target value is the reward itself
        target_values = reward + self.gamma * max_next_q_values * (1 - torch.tensor(done, dtype=torch.float, device=self.device))

        target[range(len(done)), action.argmax(dim=1)] = target_values.to(target.dtype)

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        return loss.item()