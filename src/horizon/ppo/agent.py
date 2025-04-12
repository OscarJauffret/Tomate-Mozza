import os
import torch
import time
import random

from tminterface.interface import TMInterface
from torch import Tensor

from .model import Actor, Critic, Trainer
from .rollout_buffer import RolloutBuffer
from ..agent import Agent
from ...config import Config
from ..game_interaction import send_input

class PPOAgent(Agent):
    def __init__(self, shared_dict) -> None:
        """
        Constructor for the PPOAgent class
        :param shared_dict: a dictionary to share data between threads
        """
        super().__init__(shared_dict, "PPO")

        self.hyperparameters = Config.PPO.get_hyperparameters()
        self.actor: Actor = Actor().to(self.device)
        self.critic: Critic = Critic().to(self.device)
        self.trainer: Trainer = Trainer(self.actor, self.critic, self.device, self.hyperparameters["learning_rate"],
                                        self.hyperparameters["gamma"], self.hyperparameters["lambda"],
                                        self.hyperparameters["epochs"], self.hyperparameters["epsilon"],
                                        self.hyperparameters["c1"], self.hyperparameters["c2"])

        self.memory: RolloutBuffer = RolloutBuffer(self.device)

        self.actor.train()
        self.critic.train()

    def load_model(self) -> None:
        """
        Load the model from the path chosen by the user
        :return: None
        """
        if self.shared_dict["model_path"].qsize() > 0:
            path = self.shared_dict["model_path"].get()
            actor_pth = os.path.join(path, Config.Paths.ACTOR_FILE_NAME)
            critic_pth = os.path.join(path, Config.Paths.CRITIC_FILE_NAME)
            if os.path.exists(actor_pth) and os.path.exists(critic_pth):
                self.hyperparameters = self.load_hyperparameters(path)
                self.actor.load_state_dict(torch.load(actor_pth, map_location=self.device))
                self.critic.load_state_dict(torch.load(critic_pth, map_location=self.device))
                self.trainer: Trainer = Trainer(self.actor, self.critic, self.device, self.hyperparameters["learning_rate"],
                                                self.hyperparameters["gamma"], self.hyperparameters["lambda"],
                                                self.hyperparameters["epochs"], self.hyperparameters["epsilon"],
                                                self.hyperparameters["c1"], self.hyperparameters["c2"])
                self.logger.load(os.path.join(path, Config.Paths.STAT_FILE_NAME))
                print(f"Model loaded from {path}")
            else:
                print(f"Model not found at {path}")
        else:
            # Load fresh model with random weights
            self.hyperparameters = Config.PPO.get_hyperparameters()
            self.actor: Actor = Actor().to(self.device)
            self.critic: Critic = Critic().to(self.device)
            self.trainer: Trainer = Trainer(self.actor, self.critic, self.device, self.hyperparameters["learning_rate"],
                                            self.hyperparameters["gamma"], self.hyperparameters["lambda"],
                                            self.hyperparameters["epochs"], self.hyperparameters["epsilon"],
                                            self.hyperparameters["c1"], self.hyperparameters["c2"])
            print("Loaded a fresh model with random weights")

    def save_model(self, directory) -> None:
        """
        Save the model to the path chosen by the user
        :param directory: the directory to save the model
        :return: None
        """
        actor_path = os.path.join(directory, Config.Paths.ACTOR_FILE_NAME)
        critic_path = os.path.join(directory, Config.Paths.CRITIC_FILE_NAME)
        self.actor.save_checkpoint(actor_path)
        self.critic.save_checkpoint(critic_path)

    def get_action(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Get the action from the actor and the value from the critic
        :param state: the state of the environment
        :return: the action, the log probability of the action and the value of the state
        """
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def remember(self, state: Tensor, action: Tensor, log_prob: Tensor, reward: Tensor, done: Tensor, value: Tensor) -> None:
        """
        Store the state, action, log probability, reward, done and value in the rollout buffer
        :param state: The state of the environment
        :param action: The action taken
        :param log_prob: The log probability of the action taken
        :param reward: The reward received
        :param done: Whether the episode is done
        :param value: The value of the state
        :return: None
        """
        self.memory.add(state, action, log_prob, reward, done, value)

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        if _time == 20:
            if Config.Game.RANDOM_SPAWN:
                iface.execute_command(f"load_state {random.choice(self.random_states)}")
            self.ready = True

        if _time >= 0 and _time % Config.Game.INTERVAL_BETWEEN_ACTIONS == 0 and self.ready:
            start_time = time.time()
            simulation_state = iface.get_simulation_state()
            self.agent_position.update((simulation_state.position[0], simulation_state.position[2]))

            future_state = self.thread_pool.submit(self.update_state, simulation_state)
            future_done = self.thread_pool.submit(self.determine_done, simulation_state)
            future_reward = self.thread_pool.submit(self.get_reward, simulation_state)

            future_state.result()
            done = future_done.result()
            current_reward = future_reward.result()
            self.reward += current_reward.item()

            action, log_probs, value = self.get_action(self.current_state)
            self.remember(self.current_state, action, log_probs, current_reward, done, value)

            self.prev_positions.append((simulation_state.position[0], simulation_state.position[2]))

            send_input(iface, action.item())                # Send the action to the game

            warn = True
            if self.memory.is_full():
                warn = False
                iface.set_speed(0)
                self.trainer.train_step(self.memory)
                self.memory.clear()
                iface.set_speed(self.game_speed)

            end_time = time.time()
            total_time = end_time - start_time
            if warn and total_time * 1000 > Config.Game.INTERVAL_BETWEEN_ACTIONS / self.game_speed:
                print(f"Warning: the action took {total_time * 1000:.2f}ms to execute, it should've taken less than {Config.Game.INTERVAL_BETWEEN_ACTIONS / self.game_speed:.2f}ms")

            if done:
                self.ready = False
                self.reset(iface, _time)

