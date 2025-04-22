import os
import torch
import numpy as np
import random
import time

from tminterface.interface import TMInterface

from .model import Model, Trainer
from .n_step_buffer import NStepBuffer
from .prioritized_replay_buffer import PrioritizedReplayBuffer
from ..game_interaction import send_input, launch_map
from ..agent import Agent
from ...config import Config
from ...utils.utils import save_pb

class DQNAgent(Agent):
    def __init__(self, shared_dict) -> None:
        super(DQNAgent, self).__init__(shared_dict, "DQN")

        self.hyperparameters = Config.DQN.get_hyperparameters()
        self.model: Model = Model(self.device, Config.DQN.NUMBER_OF_QUANTILES, Config.DQN.N_COS, Config.DQN.ENABLE_NOISY_NETWORK,
                                  Config.DQN.ENABLE_DUELING_NETWORK).to(self.device)
        self.trainer: Trainer = Trainer(self.model, self.device, self.hyperparameters["learning_rate"], self.hyperparameters["gamma"])
        self.memory: PrioritizedReplayBuffer = PrioritizedReplayBuffer(self.hyperparameters["max_memory"], alpha=self.hyperparameters["alpha"],
                                                                       beta=self.hyperparameters["beta_start"], device=self.device)

        self.n_step_buffer: NStepBuffer = NStepBuffer(self.hyperparameters["n_steps"], self.device)
        self.epsilon = self.hyperparameters["epsilon_start"]

        self.model.train()


    def load_model(self) -> None:
        """
        Load the model from the path chosen by the user
        :return: None
        """
        if self.shared_dict["model_path"].value:
            path = self.shared_dict["model_path"].value
            self.logger.set_directory(path)
            model_pth = os.path.join(path, Config.Paths.DQN_MODEL_FILE_NAME)
            buffer_path = os.path.join(path, Config.Paths.DQN_REPLAY_FILE_NAME)
            if os.path.exists(model_pth):
                self.hyperparameters = self.load_hyperparameters(path)
                self.model = Model(self.device, self.hyperparameters["number_of_quantiles"], self.hyperparameters["n_cos"],
                                  self.hyperparameters["enable_noisy_network"], self.hyperparameters["enable_dueling_network"]).to(self.device)
                self.model.load_state_dict(torch.load(model_pth, map_location=self.device))
                self.setup_training()
                self.logger.load(path)

                if os.path.exists(buffer_path):
                    try:
                        buffer_state = torch.load(buffer_path, map_location=self.device)

                        max_idx = min(buffer_state["fill_level"], self.memory.capacity)
                        self.memory.states[:max_idx] = buffer_state["states"][:max_idx]
                        self.memory.actions[:max_idx] = buffer_state["actions"][:max_idx]
                        self.memory.rewards[:max_idx] = buffer_state["rewards"][:max_idx]
                        self.memory.next_states[:max_idx] = buffer_state["next_states"][:max_idx]
                        self.memory.dones[:max_idx] = buffer_state["dones"][:max_idx]
                        self.memory.priorities[:max_idx] = buffer_state["priorities"][:max_idx]

                        self.memory.pos = buffer_state["pos"] if buffer_state["pos"] < self.memory.capacity else 0
                        self.memory.fill_level = min(buffer_state["fill_level"], self.memory.capacity)
                        self.memory.beta = buffer_state["beta"]

                        print(f"Replay buffer loaded")
                    except Exception as e:
                        print(f"Error loading the replay buffer: {e}")
                else:
                    print("No replay buffer found, starting with an empty one")
                print(f"Model loaded from {model_pth}")
            else:
                print(f"Model not found at {model_pth}")
        else:
            self.logger.set_directory(None)
            # Load fresh model with random weights
            self.hyperparameters = Config.DQN.get_hyperparameters()
            self.model = Model(self.device, Config.DQN.NUMBER_OF_QUANTILES, Config.DQN.N_COS, Config.DQN.ENABLE_NOISY_NETWORK,
                                  Config.DQN.ENABLE_DUELING_NETWORK).to(self.device)
            self.setup_training()
            print("Loaded a fresh model with random weights")

    def save_model(self, directory) -> None:
        """
        Save the model to the path chosen by the user
        :param directory: the directory to save the model to
        :return: None
        """
        model_path = os.path.join(directory, Config.Paths.DQN_MODEL_FILE_NAME)
        torch.save(self.model.state_dict(), model_path)

        # Save the replay buffer
        buffer_path = os.path.join(directory, Config.Paths.DQN_REPLAY_FILE_NAME)

        buffer_state = {
            "states": self.memory.states[:self.memory.fill_level],
            "actions": self.memory.actions[:self.memory.fill_level],
            "rewards": self.memory.rewards[:self.memory.fill_level],
            "next_states": self.memory.next_states[:self.memory.fill_level],
            "dones": self.memory.dones[:self.memory.fill_level],
            "priorities": self.memory.priorities[:self.memory.fill_level],
            "pos": self.memory.pos,
            "fill_level": self.memory.fill_level,
            "alpha": self.memory.alpha,
            "beta": self.memory.beta,
        }

        print(f"Saving replay buffer")
        torch.save(buffer_state, buffer_path)


    def setup_training(self) -> None:
        """
        Setup the training parameters: Trainer, n_step_buffer, and memory
        :return: None
        """
        self.trainer = Trainer(self.model, self.device, self.hyperparameters["learning_rate"],
                               self.hyperparameters["gamma"])
        self.n_step_buffer = NStepBuffer(self.hyperparameters["n_steps"], self.device)
        self.memory = PrioritizedReplayBuffer(self.hyperparameters["max_memory"], alpha=self.hyperparameters["alpha"],
                                              beta=self.hyperparameters["beta_start"], device=self.device)

    def update_epsilon(self) -> None:
        """
        Update the epsilon value for the epsilon-greedy policy
        :return: None
        """
        if self.eval:
            self.epsilon = 0
        else:
            self.epsilon = self.hyperparameters["epsilon_end"] + \
                           (self.hyperparameters["epsilon_start"] - self.hyperparameters["epsilon_end"]) \
                           * np.exp(-1. * self.iterations / self.hyperparameters["epsilon_decay"])

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get the action from the model using epsilon-boltzmann policy
        :param state: the state of the environment
        :return: the action
        """
        self.update_epsilon()

        if random.random() < self.epsilon:
            # Epsilon-greedy policy
            action = torch.randint(0, Config.Arch.OUTPUT_SIZE, (), device=self.device)
            return action
        else:
            with torch.no_grad():
                prediction = self.model(state.unsqueeze(0)) # Shape: (1, n_quantiles, n_actions)
                expected_q = torch.mean(prediction, dim=1)  # Shape: (1, n_actions)
                action = torch.argmax(expected_q, dim=1)
                if self.eval:
                    for i, output in enumerate(Config.Arch.OUTPUTS_DESC):
                        self.shared_dict["q_values"][output] = expected_q.squeeze(0)[i].item()
                    self.shared_dict["q_values"]["is_random"] = False
                return action


    def remember(self, state, action, reward, next_state, done) -> None:
        """
        Store the state, action, reward, next state and done in the memory
        :param state: The state of the environment
        :param action: The action taken
        :param reward: The reward received
        :param next_state: The next state of the environment
        :param done: Whether the episode is done
        :return: None
        """
        self.memory.add((state, action, reward, next_state, done))  # No priority, it will take the highest priority by default

    def train_long_memory(self) -> None:
        """
        Train the model
        :return: None
        """
        batch = self.memory.sample(self.hyperparameters["batch_size"])
        if batch is None:
            return

        (states, actions, rewards, next_states, dones), indices, weights = batch

        td_sample = self.trainer.train_step(states, actions, rewards, next_states, dones, weights)
        self.memory.update_priorities(indices, td_sample)


    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        if _time == 0:
            if Config.Game.RANDOM_SPAWN:
                self.spawn_point = random.randint(0, len(self.unlocked_states))
                iface.execute_command(f"load_state {self.random_states[self.spawn_point]}")
            self.ready = True
            if self.save_pb:
                self.save_pb = False
                save_pb(self.shared_dict["model_path"].value, self.previous_finish_time, self.spawn_point != 0) # Save the pb only if not random spawn
                launch_map(iface)
                return


        if _time >= 0 and _time % Config.Game.INTERVAL_BETWEEN_ACTIONS == 0 and self.ready:
            start_time = time.time()
            simulation_state = iface.get_simulation_state()
            self.agent_position.update((simulation_state.position[0], simulation_state.position[2]))
            self.update_state(simulation_state)  # Get the current state
            done = self.determine_done(simulation_state)
            current_reward = 0
            if len(self.n_step_buffer) > 0:
                current_reward = self.get_reward(simulation_state)
                self.reward += current_reward.item()

            action = self.get_action(self.current_state)
            if not self.eval:
                self.n_step_buffer.add(self.current_state, action, current_reward)
            self.prev_positions.append((simulation_state.position[0], simulation_state.position[2]))

            #send_input(iface, action.item())  # Send the action to the game

            if self.n_step_buffer.is_full() and not self.eval:
                state, action, reward = self.n_step_buffer.get_transition()
                next_state = self.current_state
                self.remember(state.clone(), action.clone(), reward, next_state, done)

            end_time = time.time()
            total_time = end_time - start_time
            if total_time * 1000 > Config.Game.INTERVAL_BETWEEN_ACTIONS / self.game_speed:
                print(f"Warning: the action took {total_time * 1000:.2f}ms to execute, it should've taken less than {Config.Game.INTERVAL_BETWEEN_ACTIONS / self.game_speed:.2f}ms")

            if done:
                self.ready = False

                if not self.eval:
                    while not self.n_step_buffer.is_empty():
                        state, action, reward = self.n_step_buffer.get_transition()
                        next_state = self.current_state
                        self.n_step_buffer.pop_transition()
                        self.remember(state.clone(), action.clone(), reward, next_state, done)
                    self.train_long_memory()
                    if self.iterations % Config.DQN.UPDATE_TARGET_EVERY == 0:
                        self.trainer.update_target()

                self.n_step_buffer.clear()
                self.reset(iface, _time)
                iface.set_speed(self.game_speed)



