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
        x = self.layer3(x)
        return x

class QTrainer:
    def __init__(self, model, device, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        self.criterion = nn.SmoothL1Loss()  # Huber loss

    def train_step(self, state, action, reward, next_state, done):
        state = state.to(self.device) if state.device != self.device else state
        next_state = next_state.to(self.device) if next_state.device != self.device else next_state
        action = action.to(self.device) if action.device != self.device else action
        reward = reward.to(self.device) if reward.device != self.device else reward
        done = done.to(self.device) if done.device != self.device else done

        done = done.type(torch.float)       # FIXME: maybe use float from the beginning

        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # Predicted Q values
        pred = self.model(state)        # Predicted Q values**
        action_indexes = torch.argmax(action, dim=1)    # Indexes of the actions taken (it is plural because we might have a batch of actions)
        target = pred.clone()

        with torch.no_grad():
            next_q_value = self.model(next_state).max(dim=1).values  # This gives us the maximum reward we can get in the next state as a 1D tensor (ex: if batch of size 1, tensor([0.5]), if batch of size 2, tensor([0.5, 0.6]))
        target[range(len(action_indexes)), action_indexes] = (reward + self.gamma * next_q_value * (1 - done)).type(torch.float)

        self.optimizer.zero_grad(set_to_none=True)
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()