import os.path
import random

import numpy as np
import json

from tminterface.client import Client
from tminterface.interface import TMInterface
from tminterface.structs import SimStateData
from collections import deque
from torch import Tensor
import shutil

from .ppomodel import PPOActor, PPOCritic, PPOTrainer
from .rollout_buffer import RolloutBuffer
from ..map_interaction.agent_position import AgentPosition
from ..utils.utils import *
from ..config import Config
from ..utils.tm_logger import TMLogger

class HorizonClient(Client):
    def __init__(self, shared_dict) -> None:
        super(HorizonClient, self).__init__()
        self.agent_position = AgentPosition()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.hyperparameters = Config.NN.get_hyperparameters()
        self.actor: PPOActor = PPOActor().to(self.device)
        self.critic: PPOCritic = PPOCritic().to(self.device)
        self.trainer: PPOTrainer = PPOTrainer(self.actor, self.critic, self.device)

        self.memory: RolloutBuffer = RolloutBuffer(self.device)
        self.reward = 0.0

        self.prev_positions = deque(maxlen=5 * Config.Game.NUMBER_OF_ACTIONS_PER_SECOND)        # 5-second long memory
        self.current_state = None
        self.prev_velocity = None
        self.has_finished = False

        self.iterations = 0
        self.ready = False

        self.logger = TMLogger(get_device_info(self.device.type))
        self.rewards_queue = shared_dict["reward"]
        self.eval_mode = shared_dict["eval"]
        self.q_values_dict = shared_dict["q_values"]
        self.model_path = shared_dict["model_path"]
        self.manual_epsilon = None

        if not os.path.exists(os.path.join(get_states_path(), Config.Paths.MAP)):
            print(f"Map directory not found at {os.path.join(get_states_path(), Config.Paths.MAP)}")
            self.random_states = []
        else:
            self.random_states = [os.path.join(Config.Paths.MAP, state) for state in os.listdir(os.path.join(get_states_path(), Config.Paths.MAP)) if state.endswith(".bin")]


    def __str__(self) -> str:
        return (f"x position: {self.current_state[0].item():<8.2f} y position: {self.current_state[1].item():<8.2f} next turn: {self.current_state[2].item():<8} "
                f"velocity: {self.current_state[3].item():<8.2f} acceleration: {self.current_state[4].item():<8.2f} relative yaw: {self.current_state[5].item():<8.2f} "
                f"next edge length: {self.current_state[6].item():<8.2f} second turn: {self.current_state[7].item():<8} third edge length: {self.current_state[8].item():<8.2f} "
                f"third turn: {self.current_state[9].item():<8}")

#    def load_model(self) -> None:
#        if self.model_path.qsize() > 0:
#            path = self.model_path.get()
#            model_pth = os.path.join(path, Config.Paths.MODEL_FILE_NAME)
#            if os.path.exists(model_pth):
#                self.hyperparameters = self.load_hyperparameters(path)
#                self.model.load_state_dict(torch.load(model_pth, map_location=self.device))
#                self.trainer = QTrainer(self.model, self.device, self.hyperparameters["learning_rate"], self.hyperparameters["gamma"])
#                self.logger.load(os.path.join(path, Config.Paths.STAT_FILE_NAME))
#                print(f"Model loaded from {model_pth}")
#            else:
#                print(f"Model not found at {model_pth}")
#        else:
#            # Load fresh model with random weights
#            self.hyperparameters = Config.NN.get_hyperparameters()
#            self.model = DQNModel().to(self.device)
#            self.trainer = QTrainer(self.model, self.device, self.hyperparameters["learning_rate"], self.hyperparameters["gamma"])
#            print("Loaded a fresh model with random weights")

    def load_hyperparameters(self, path: str) -> dict:
        hyperparameters = {}
        hyperparameters_path = os.path.join(path, Config.Paths.STAT_FILE_NAME)
        if os.path.exists(hyperparameters_path):
            with open(hyperparameters_path, "r") as f:
                stats = json.load(f)
                hyperparameters = stats["hyperparameters"]
                self.iterations = stats["statistics"]["total number of runs"]
        return hyperparameters

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")

    def launch_map(self, iface: TMInterface) -> None:
        iface.execute_command(f"map {get_default_map()}")
        iface.set_speed(Config.Game.GAME_SPEED)

    def save_model(self) -> None:
        self.logger.update_log_id()
        directory = self.logger.dump()

        if not directory:
            print("Failed to save the model")
            return

        actor_path = os.path.join(directory, Config.Paths.ACTOR_FILE_NAME)
        critic_path = os.path.join(directory, Config.Paths.CRITIC_FILE_NAME)
        self.actor.save_checkpoint(actor_path)
        self.critic.save_checkpoint(critic_path)
        
        # Copy the model dir contents to the latest model dir
        for file in os.listdir(directory):
            shutil.copy(os.path.join(directory, file), Config.Paths.LATEST_MODEL_PATH)

        print(f"Model saved to {directory} and in the latest model directory")

    def get_state(self, simulation_state: SimStateData) -> Tensor:
        current_velocity = np.array(simulation_state.velocity)

        acceleration_scalar = 0
        velocity_norm = 0
        if self.prev_velocity is not None:
            delta_v = current_velocity - self.prev_velocity
            velocity_norm = np.linalg.norm(current_velocity)
            if velocity_norm > 1e-5:
                direction = current_velocity / velocity_norm
                acceleration_scalar = np.dot(delta_v, direction) / (Config.Game.INTERVAL_BETWEEN_ACTIONS / 1000)
        self.prev_velocity = current_velocity

        agent_absolute_position = (simulation_state.position[0], simulation_state.position[2])
        section_relative_position, next_turn, second_edge_length, second_turn, third_edge_length, third_turn = self.agent_position.get_relative_position_and_next_turns(agent_absolute_position)
        relative_yaw = self.agent_position.get_car_orientation(simulation_state.yaw_pitch_roll[0], agent_absolute_position)

        current_state = torch.tensor([
            section_relative_position[0],
            section_relative_position[1],
            next_turn,
            velocity_norm / 50,
            acceleration_scalar / 15,
            relative_yaw,
            second_edge_length,
            second_turn,
            third_edge_length,
            third_turn
        ], dtype=torch.float, device=self.device)

        return current_state

    def get_action(self, state: Tensor) -> tuple[int, float, Tensor]:
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        prob = dist.log_prob(action)

        return action, prob, value

    def send_input(self, iface: TMInterface, move) -> None:
        match move: 
            case 0:
                iface.set_input_state(accelerate=False, left=False, right=False)
            case 1:
                iface.set_input_state(accelerate=True, left=False, right=False)
            case 2:
                iface.set_input_state(accelerate=False, left=False, right=True)
            case 3:
                iface.set_input_state(accelerate=False, left=True, right=False)
            case 4:
                iface.set_input_state(accelerate=True, left=False, right=True)
            case 5:
                iface.set_input_state(accelerate=True, left=True, right=False)
            case _:
                iface.set_input_state(accelerate=False, left=False, right=False)

    def get_reward(self, simulation_state: SimStateData) -> torch.Tensor:
        if not self.prev_positions:
            return torch.tensor(0, device=self.device)
        prev_position = self.prev_positions[-1]
        current_position = (simulation_state.position[0], simulation_state.position[2])
        current_reward = self.agent_position.get_distance_reward(prev_position, current_position)
        if self.has_finished:
            current_reward += Config.Game.BLOCK_SIZE
        return torch.tensor(current_reward, device=self.device)

    def determine_done(self, simulation_state: SimStateData) -> torch.Tensor:

        if simulation_state.position[1] < 23: # If the car is below the track
            return torch.tensor(1.0, device=self.device, dtype=torch.float)

        if simulation_state.player_info.race_finished:
            self.has_finished = True
            return torch.tensor(1.0, device=self.device, dtype=torch.float)

        if self.prev_positions and len(self.prev_positions) == 50 and np.linalg.norm(np.array(self.prev_positions[0]) - np.array(self.prev_positions[-1])) < 5:    # If less than 5 meters were travelled in the last 5 seconds
            return torch.tensor(1.0, device=self.device, dtype=torch.float)

        return torch.tensor(0.0, device=self.device, dtype=torch.float)

    def remember(self, state, action, probs, reward, done, value):
        self.memory.add(state, action, probs, reward, done, value)

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        if _time == 20:
            if Config.Game.RANDOM_SPAWN:
                iface.execute_command(f"load_state {random.choice(self.random_states)}")
            self.ready = True

        if _time >= 0 and _time % Config.Game.INTERVAL_BETWEEN_ACTIONS == 0 and self.ready:
            start_time = time.time()
            simulation_state = iface.get_simulation_state()
            self.current_state = self.get_state(simulation_state)  # Get the current state
            done = self.determine_done(simulation_state)

            current_reward = self.get_reward(simulation_state)
            self.reward += current_reward.item()

            action, probs, value = self.get_action(self.current_state)
            self.remember(self.current_state, action, probs, current_reward, done, value)

            self.prev_positions.append((simulation_state.position[0], simulation_state.position[2]))

            self.send_input(iface, action)                # Send the action to the game

            if self.memory.is_full():
                self.trainer.train_step(self.memory)
                self.memory.clear()

            end_time = time.time()
            total_time = end_time - start_time
            if total_time * 1000 > Config.Game.INTERVAL_BETWEEN_ACTIONS / Config.Game.GAME_SPEED:
                print(f"Warning: the action took {total_time * 1000:.2f}ms to execute, it should've taken less than {Config.Game.INTERVAL_BETWEEN_ACTIONS / Config.Game.GAME_SPEED:.2f}ms")

            if done:
                self.ready = False
                if not self.eval_mode:
                    self.iterations += 1
                    self.rewards_queue.put(self.reward)
                    self.logger.add_run(self.iterations, _time, self.reward)

                print(f"Iteration: {self.iterations:<8} reward: {self.reward:<8.2f}")
                self.reward = 0.0
                self.prev_positions.clear()
                self.prev_velocity = None
                if self.has_finished:
                    self.has_finished = False
                    self.launch_map(iface)
                else:
                    iface.horn()
                    iface.execute_command(f"load_state {self.random_states[0]}")

