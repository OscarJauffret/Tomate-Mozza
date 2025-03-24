import random
import numpy as np
import torch
import json, os

from tminterface.client import Client
from tminterface.interface import TMInterface
from collections import deque
import shutil

from ..map_interaction.agent_position import AgentPosition
from ..utils.utils import *
from .model import Model, QTrainer
from ..config import Config
from ..utils.tm_logger import TMLogger

class HorizonClient(Client):
    def __init__(self, num, shared_dict) -> None:
        super(HorizonClient, self).__init__()
        self.num = num
        self.agent_position = AgentPosition()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.hyperparameters = Config.NN.get_hyperparameters()
        self.model = Model().to(self.device)
        self.trainer = QTrainer(self.model, self.device, self.hyperparameters["learning_rate"], self.hyperparameters["gamma"])

        self.memory = deque(maxlen=self.hyperparameters["max_memory"])
        self.reward = 0.0
        self.epsilon = self.hyperparameters["epsilon_start"]

        self.prev_position = None
        self.prev_positions = deque(maxlen=5 * Config.Game.NUMBER_OF_ACTIONS_PER_SECOND)        # 5-second long memory
        self.current_state = None
        self.prev_game_state: StateAction = None

        self.iterations = 0
        self.ready = False

        self.logger = TMLogger(get_device_info(self.device.type))
        self.rewards_queue = shared_dict["reward"]
        self.epsilon_dict = shared_dict["epsilon"]
        self.q_values_dict = shared_dict["q_values"]
        self.model_path = shared_dict["model_path"]
        self.manual_epsilon = None

        self.random_states = [os.path.join(Config.Paths.MAP, state) for state in os.listdir(os.path.join(get_states_path(), Config.Paths.MAP)) if state.endswith(".bin")]

        self.model.train()

    def __str__(self) -> str:
        return f"x position: {self.current_state[0].item():<8.2f} y position: {self.current_state[1].item():<8.2f} next turn: {self.current_state[2].item():<8} yaw: {self.current_state[3].item():<8.2f}"

    def load_model(self) -> None:
        if self.model_path.qsize() > 0:
            path = self.model_path.get()
            model_pth = os.path.join(path, Config.Paths.MODEL_FILE_NAME)
            if os.path.exists(model_pth):
                self.hyperparameters = self.load_hyperparameters(path)
                self.model.load_state_dict(torch.load(model_pth, map_location=self.device))
                self.trainer = QTrainer(self.model, self.device, self.hyperparameters["learning_rate"], self.hyperparameters["gamma"])
                self.logger.load(os.path.join(path, Config.Paths.STAT_FILE_NAME))
                print(f"Model loaded from {model_pth}")
            else:
                print(f"Model not found at {model_pth}")
        else:
            # Load fresh model with random weights
            self.hyperparameters = Config.NN.get_hyperparameters()
            self.model = Model().to(self.device)
            self.trainer = QTrainer(self.model, self.device, self.hyperparameters["learning_rate"], self.hyperparameters["gamma"])
            print("Loaded a fresh model with random weights")

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

        model_path = os.path.join(directory, Config.Paths.MODEL_FILE_NAME)
        torch.save(self.model.state_dict(), model_path)
        
        # Copy the model dir contents to the latest model dir
        for file in os.listdir(directory):
            shutil.copy(os.path.join(directory, file), Config.Paths.LATEST_MODEL_PATH)

        print(f"Model saved to {model_path} and in the latest model directory")

    def get_state(self, iface: TMInterface):
        state = iface.get_simulation_state()

        current_state = torch.tensor([
            0.0,
            0.0,
            0.0,
            state.display_speed / 999,
            0.0
        ], dtype=torch.float, device=self.device)

        return current_state
    
    def update_epsilon(self):
        if self.epsilon_dict["manual"]:
            self.epsilon = self.epsilon_dict["value"]
        else:
            self.epsilon = self.hyperparameters["epsilon_end"] + \
                (self.hyperparameters["epsilon_start"] - self.hyperparameters["epsilon_end"]) \
                      * np.exp(-1. * self.iterations / self.hyperparameters["epsilon_decay"])
            self.epsilon_dict["value"] = self.epsilon

    def get_action(self, state) -> torch.Tensor:
        self.update_epsilon()

        if random.random() < self.epsilon:
            move = torch.randint(0, Config.NN.Arch.OUTPUT_SIZE, (), device=self.device)
            for i, action in enumerate(Config.NN.Arch.OUTPUTS_DESC):
                self.q_values_dict[action] = 0.0
            self.q_values_dict[Config.NN.Arch.OUTPUTS_DESC[move]] = 1.0
            self.q_values_dict["is_random"] = True
            return move
        else:
            with torch.no_grad():
                prediction = self.model(state.to(self.device))
                for i, action in enumerate(Config.NN.Arch.OUTPUTS_DESC):
                    self.q_values_dict[action] = prediction[i].item()
                self.q_values_dict["is_random"] = False
                return torch.argmax(prediction)


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

    def get_reward(self, iface: TMInterface):
        if self.prev_position is None:
            return torch.tensor(0, device=self.device)
        prev_position = self.prev_position
        current_position = iface.get_simulation_state().position[0], iface.get_simulation_state().position[2]
        current_reward = self.agent_position.get_distance_reward(prev_position, current_position)
        # print(f"Reward: {current_reward}")
        return torch.tensor(current_reward, device=self.device)

    def determine_done(self, iface: TMInterface):
        state = iface.get_simulation_state()

        if state.position[1] < 23: # If the car is below the track
            return torch.tensor(1.0, device=self.device, dtype=torch.float)
        if not state.player_info.finish_not_passed:
            return torch.tensor(1.0, device=self.device, dtype=torch.float)
        # if self.prev_positions and len(self.prev_positions) == 50 and np.linalg.norm(np.array(self.prev_positions[0]) - np.array(self.prev_positions[-1])) < 5:    # If less than 5 meters were travelled in the last 5 seconds
        #     return torch.tensor(1.0, device=self.device, dtype=torch.float)
        #if state.display_speed < 5 and state.race_time > 2000:
        #    return torch.tensor(True, device=self.device)
        return torch.tensor(0.0, device=self.device, dtype=torch.float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > self.hyperparameters["batch_size"]:
            mini_sample = random.sample(self.memory, self.hyperparameters["batch_size"])
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states =  torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.stack(dones).to(self.device)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        if _time == 0:
            if Config.Game.RANDOM_SPAWN:
                iface.execute_command(f"load_state {random.choice(self.random_states)}")
            self.ready = True

        if _time >= 0 and _time % Config.Game.INTERVAL_BETWEEN_ACTIONS == 0 and self.ready:
            start_time = time.time()
            self.current_state = self.get_state(iface)  # Get the current state
            done = self.determine_done(iface)

            if self.prev_game_state is not None:         # If this is not the first state, train the model
                current_reward = self.get_reward(iface)
                if not self.epsilon_dict["manual"]:
                    self.remember(self.prev_game_state.state, self.prev_game_state.action, current_reward, self.current_state, done)
                    self.train_short_memory(self.prev_game_state.state, self.prev_game_state.action, current_reward, self.current_state, done)
                self.reward += current_reward.item()

            action = self.get_action(self.current_state)                                    # Get the action
            self.prev_game_state = StateAction(self.current_state, action)       # Save the current state and action for the next iteration
            self.prev_position = iface.get_simulation_state().position[0], iface.get_simulation_state().position[2] # Save the previous position for the reward's calculation
            self.prev_positions.append(self.prev_position)

            # self.send_input(iface, action)                # Send the action to the game

            end_time = time.time()
            total_time = end_time - start_time
            if total_time * 1000 > Config.Game.INTERVAL_BETWEEN_ACTIONS / Config.Game.GAME_SPEED:
                print(f"Warning: the action took {total_time * 1000:.2f}ms to execute, it should've taken less than {Config.Game.INTERVAL_BETWEEN_ACTIONS / Config.Game.GAME_SPEED:.2f}ms")

            if done:
                self.ready = False
                if not self.epsilon_dict["manual"]:
                    self.iterations += 1
                    self.rewards_queue.put(self.reward)
                    self.logger.add_run(self.iterations, _time, self.reward)
                    self.train_long_memory()
                print(f"Iteration: {self.iterations:<8} reward: {self.reward:<8.2f} epsilon: {self.epsilon:<8.3f}")    
                self.reward = 0.0
                self.prev_position = None
                self.prev_positions.clear()
                self.prev_game_state = None
                iface.horn()
                iface.respawn()

class StateAction:
    def __init__(self, state, action):
        self.state = state
        self.action = action