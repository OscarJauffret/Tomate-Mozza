import numpy as np

from ..config import Config

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.dones = []
        self.values = []

        self.batch_size = Config.NN.BATCH_SIZE

    def is_full(self):
        return len(self.states) >= Config.NN.MEMORY_SIZE

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state.detach())  # Each of these lists is a list of tensors.
        self.actions.append(action.detach())
        self.probs.append(log_prob.detach())
        self.rewards.append(reward.detach())
        self.dones.append(done.detach())
        self.values.append(value.detach())

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start] # batches is a list of arrays of indices. These indices indicate a batch of states, actions, probs, values, rewards and dones

        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.values), np.array(self.rewards), np.array(self.dones), batches
