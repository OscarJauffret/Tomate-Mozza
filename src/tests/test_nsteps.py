import unittest
import torch
import numpy as np
from collections import deque
from torch import Tensor

torch.manual_seed(0)
np.random.seed(0)

class DummyConfig:
    class NN:
        GAMMA = 0.99
        MAX_MEMORY: int = 100
        MIN_MEMORY: int = 10
        BATCH_SIZE: int = 5
        BETA_START: float = 0.4
        BETA_MAX: float = 1.0
        BETA_INCREMENT_STEPS: int = 40000
        class Arch:
            INPUT_SIZE = 10
            OUTPUT_SIZE = 6
        


Config = DummyConfig()

class NStepBufferOld:
    def __init__(self, n_steps:int) -> None:
        self.n_steps = n_steps
        self.states = deque(maxlen=self.n_steps)
        self.actions = deque(maxlen=self.n_steps)
        self.rewards = deque(maxlen=self.n_steps)
        self.gammas = Config.NN.GAMMA ** np.arange(self.n_steps)

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def get_transition(self) -> tuple[Tensor, int, float]:
        return self.states[0], self.actions[0], self.cumulative_reward()
    
    def pop_transition(self):
        self.states.popleft()
        self.actions.popleft()
        self.rewards.popleft()

    def is_full(self):
        return len(self.states) == self.n_steps

    def is_empty(self):
        return len(self.states) == 0

    def cumulative_reward(self):
        return sum([self.rewards[i] * self.gammas[i] for i in range(len(self.rewards))])

    def add(self, state: Tensor, action: int, reward: float):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    ...

class NStepBufferNew:
    def __init__(self, n_steps:int, device) -> None:
        self.n_steps = n_steps
        self.current_size = 0
        self.position = 0
        self.device = device

        self.states = torch.zeros((self.n_steps, Config.NN.Arch.INPUT_SIZE), dtype=torch.float, device=self.device)
        self.actions = torch.zeros(self.n_steps, dtype=torch.int64, device=self.device)
        self.rewards = torch.zeros(self.n_steps, dtype=torch.float, device=self.device)
        self.gammas = torch.tensor(Config.NN.GAMMA ** np.arange(self.n_steps), dtype=torch.float, device=self.device)

    def __len__(self):
        return self.current_size

    def clear(self):
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()

        self.current_size = 0
        self.position = 0

    
    def get_transition(self) -> tuple[Tensor, Tensor, Tensor]:
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

    def add(self, state: Tensor, action: Tensor, reward: Tensor):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward

        self.position = (self.position + 1) % self.n_steps
        if self.current_size < self.n_steps:
            self.current_size += 1

class TestNStepNewBufferEquivalence(unittest.TestCase):
    def setUp(self):
        self.n_steps = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bufferOld = NStepBufferOld(self.n_steps)
        self.bufferNew = NStepBufferNew(self.n_steps, self.device)

        n_transitions = int(self.n_steps * 1.6)

        self.states = [torch.randn(Config.NN.Arch.INPUT_SIZE, device=self.device) for _ in range(n_transitions)]
        self.actions = [torch.randint(0, Config.NN.Arch.OUTPUT_SIZE, (1,), device=self.device) for _ in range(n_transitions)]
        self.rewards = [torch.rand(1, device=self.device) for _ in range(n_transitions)]

    def test_equivalence(self):
        for state, action, reward in zip(self.states, self.actions, self.rewards):
            self.bufferOld.add(state, action.item(), reward.item())
            self.bufferNew.add(state, action, reward)

        stateOld, actionOld, R_Old = self.bufferOld.get_transition()
        stateNew, actionNew, R_New = self.bufferNew.get_transition()

        # print("Old buffer state:", stateOld)
        # print("New buffer state:", stateNew)
        # print("Old buffer action:", actionOld)
        # print("New buffer action:", actionNew)
        # print("Old buffer cumulative reward:", R_Old)
        # print("New buffer cumulative reward:", R_New)


        # Test état
        self.assertTrue(torch.allclose(stateOld, stateNew, atol=1e-5), "States are not equal")
        
        # Test action
        self.assertEqual(actionOld, actionNew.item(), "Actions are not equal")
        
        # Test reward cumulée
        self.assertAlmostEqual(R_Old, R_New.item(), places=5, msg="Cumulative rewards differ")

        while not self.bufferOld.is_empty():
            old = self.bufferOld.get_transition()
            new = self.bufferNew.get_transition()
            self.assertTrue(torch.allclose(old[0], new[0], atol=1e-5), "States are not equal")
            self.assertEqual(old[1], new[1].item(), "Actions are not equal")
            self.assertAlmostEqual(old[2], new[2].item(), places=5, msg="Cumulative rewards differ")

            self.bufferOld.pop_transition()
            self.bufferNew.pop_transition()
            self.assertEqual(len(self.bufferOld), len(self.bufferNew), "Lengths differ after pop")

        # Test clear
        self.bufferOld.add(self.states[0], self.actions[0].item(), self.rewards[0].item())
        self.bufferNew.add(self.states[0], self.actions[0], self.rewards[0])
        self.bufferOld.clear()
        self.bufferNew.clear()
        self.assertTrue(self.bufferOld.is_empty())
        self.assertTrue(self.bufferNew.is_empty())

    def test_multiple_times(self):
        for _ in range(5):
            self.test_equivalence()


class PrioritizedReplayBuffer:

    def __init__(self, capacity, alpha, beta):
        self.capacity = capacity
        self.buffer = np.zeros(self.capacity, dtype=object)
        self.priorities = np.zeros(self.capacity)
        self.pos = 0
        self.fill_level = 0

        self.alpha = alpha
        self.beta = beta
        self.epsilon = 0.001

    def __len__(self):
        return self.fill_level

    def add(self, transition: tuple, priority=None):
        if priority is None:
            priority = self.priorities.max() if self.fill_level > 0 else 1.0

        self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        self.fill_level = min(self.fill_level + 1, self.capacity)

    def sample(self, batch_size):
        if self.fill_level < Config.NN.MIN_MEMORY or batch_size > self.fill_level:
            return None

        torch.manual_seed(0)
        np.random.seed(0)

        self.beta += (Config.NN.BETA_MAX - Config.NN.BETA_START) / Config.NN.BETA_INCREMENT_STEPS
        self.beta = min(self.beta, Config.NN.BETA_MAX)

        valid_priorities = self.priorities[:self.fill_level]
        scaled = (valid_priorities + self.epsilon) ** self.alpha
        probabilities = scaled / scaled.sum()

        indices = np.random.choice(self.fill_level, batch_size, p=probabilities)
        samples = np.array(self.buffer, dtype=object)[indices]

        weights = ((1 / self.fill_level) * (1 / probabilities[indices])) ** self.beta
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class TestNstepBuffersWithPrioritizedReplayBuffer(unittest.TestCase):

    def setUp(self):
        self.n_steps = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bufferOld = NStepBufferOld(self.n_steps)
        self.bufferNew = NStepBufferNew(self.n_steps, self.device)

        self.prioritized_buffer_old = PrioritizedReplayBuffer(100, 0.6, 0.4)
        self.prioritized_buffer_new = PrioritizedReplayBuffer(100, 0.6, 0.4)

        n_transitions = int(self.n_steps * 1.6)

        self.states = [torch.randn(Config.NN.Arch.INPUT_SIZE, device=self.device) for _ in range(n_transitions)]
        self.actions = [torch.randint(0, Config.NN.Arch.OUTPUT_SIZE, (1,), device=self.device) for _ in range(n_transitions)]
        self.rewards = [torch.rand(1, device=self.device) for _ in range(n_transitions)]

    def test_equivalence(self):
        for state, action, reward in zip(self.states, self.actions, self.rewards):
            self.bufferOld.add(state, action.item(), reward.item())
            self.bufferNew.add(state, action, reward)

        next_state = torch.randn(Config.NN.Arch.INPUT_SIZE, device=self.device)
        done = torch.tensor(1.0, device=self.device, dtype=torch.float)

        while not self.bufferOld.is_empty():
            state_old, action_old, R_old = self.bufferOld.get_transition()
            state_new, action_new, R_new = self.bufferNew.get_transition()

            self.assertTrue(torch.allclose(state_old, state_new, atol=1e-5), "States are not equal in buffers")
            self.assertEqual(action_old, action_new.item(), "Actions are not equal in buffers")
            self.assertAlmostEqual(R_old, R_new.item(), places=5, msg="Cumulative rewards differ in buffers")

            self.prioritized_buffer_old.add((state_old, action_old, R_old, next_state, done))
            self.prioritized_buffer_new.add((state_new, action_new, R_new, next_state, done))

            self.bufferOld.pop_transition()
            self.bufferNew.pop_transition()

        self.assertTrue(self.bufferOld.is_empty(), "Old buffer is not empty")
        self.assertTrue(self.bufferNew.is_empty(), "New buffer is not empty")

        self.assertGreater(len(self.prioritized_buffer_old), Config.NN.MIN_MEMORY, "Old buffer is not full enough")
        self.assertGreater(len(self.prioritized_buffer_new), Config.NN.MIN_MEMORY, "New buffer is not full enough")

        samples_old, indices_old, weights_old = self.prioritized_buffer_old.sample(5)
        samples_new, indices_new, weights_new = self.prioritized_buffer_new.sample(5)
        self.assertIsNotNone(samples_old, "Old buffer sample is None")
        self.assertIsNotNone(samples_new, "New buffer sample is None")
        self.assertEqual(len(samples_old), len(samples_new), "Sample lengths differ")

        for old_sample, new_sample in zip(samples_old, samples_new):
            state_old, action_old, R_old, next_state_old, done_old = old_sample
            state_new, action_new, R_new, next_state_new, done_new = new_sample

            self.assertTrue(torch.allclose(state_old, state_new, atol=1e-5), "States are not equal")
            self.assertEqual(action_old, action_new.item(), "Actions are not equal")
            self.assertAlmostEqual(R_old, R_new.item(), places=5, msg="Cumulative rewards differ")
            self.assertTrue(torch.allclose(next_state_old, next_state_new, atol=1e-5), "Next states are not equal")
            self.assertEqual(done_old.item(), done_new.item(), "Done flags are not equal")

        for old_index, new_index in zip(indices_old, indices_new):
            self.assertEqual(old_index, new_index, "Indices are not equal")

        for old_weight, new_weight in zip(weights_old, weights_new):
            self.assertAlmostEqual(old_weight, new_weight.item(), places=5, msg="Weights differ")



if __name__ == '__main__':
    unittest.main()
