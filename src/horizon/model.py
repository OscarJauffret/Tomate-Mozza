import torch
import torch.nn as nn
import copy
from ..config import Config

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(Config.NN.Arch.INPUT_SIZE, Config.NN.Arch.LAYER_SIZES[0])
        self.layer2 = nn.Linear(Config.NN.Arch.LAYER_SIZES[0], Config.NN.Arch.LAYER_SIZES[1])
        self.output = nn.Linear(Config.NN.Arch.LAYER_SIZES[1], Config.NN.Arch.OUTPUT_SIZE)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x

class QTrainer:
    def __init__(self, model, device, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.main_model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        self.criterion = nn.SmoothL1Loss(reduction="none")  # Huber loss

        self.target_model = copy.deepcopy(self.main_model).to(self.device)
        self.target_model.eval()

    def train_step(self, state, action, reward, next_state, done, weights=None):
        state = state.to(self.device) if state.device != self.device else state
        next_state = next_state.to(self.device) if next_state.device != self.device else next_state
        action = action.to(self.device) if action.device != self.device else action
        reward = reward.to(self.device) if reward.device != self.device else reward
        done = done.to(self.device) if done.device != self.device else done

        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # Predicted Q values
        pred = self.main_model(state)        # Predicted Q values
        target = pred.clone()

        with torch.no_grad():
            next_action = self.main_model(next_state).argmax(dim=1)  # This gives us the action that gives us the maximum reward in the next state as a 1D tensor (ex: if batch of size 1, tensor([0]), if batch of size 2, tensor([0, 1]))
            next_q_target = self.target_model(next_state).gather(1, next_action.unsqueeze(1)).squeeze(1)
        target[range(len(action)), action] = (reward + self.gamma * next_q_target * (1 - done)).type(torch.float)

        if torch.any(target > 1000):
            print("Target values are too high")

        td_errors = (pred[range(len(action)), action] - target[range(len(action)), action]).abs().detach()

        self.optimizer.zero_grad(set_to_none=True)
        loss = self.criterion(pred, target)
        if weights is not None:
            weights = torch.tensor(weights).to(self.device)
            per_sample_loss = loss[range(len(action)), action]
            loss = (per_sample_loss * weights).mean()

        loss.backward()
        self.optimizer.step()

        return td_errors.cpu().numpy()

    def update_target(self):
        self.target_model.load_state_dict(self.main_model.state_dict())