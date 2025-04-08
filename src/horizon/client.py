import os.path
import random

import numpy as np
import json

from tminterface.client import Client
from tminterface.interface import TMInterface
from tminterface.structs import SimStateData
from collections import deque
import shutil

from ..map_interaction.agent_position import AgentPosition
from ..utils.utils import *
from .model import Model, QTrainer
from ..config import Config
from ..utils.tm_logger import TMLogger
from .prioritized_replay_buffer import PrioritizedReplayBuffer

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

        self.memory = PrioritizedReplayBuffer(self.hyperparameters["max_memory"], alpha=self.hyperparameters["alpha"], beta=self.hyperparameters["beta_start"]) 
        self.reward = 0.0
        self.epsilon = self.hyperparameters["epsilon_start"]

        self.prev_position = None
        self.prev_positions = deque(maxlen=5 * Config.Game.NUMBER_OF_ACTIONS_PER_SECOND)        # 5-second long memory
        self.current_state = torch.zeros(Config.NN.Arch.INPUT_SIZE, dtype=torch.float, device=self.device)
        self.n_step_buffer: NStepBuffer = NStepBuffer(self.hyperparameters["n_steps"], self.device)
        self.prev_velocity = None
        self.has_finished = False

        self.iterations = 0
        self.ready = False

        self.logger = TMLogger(get_device_info(self.device.type))
        self.rewards_queue = shared_dict["reward"]
        self.epsilon_dict = shared_dict["epsilon"]
        self.q_values_dict = shared_dict["q_values"]
        self.model_path = shared_dict["model_path"]
        self.manual_epsilon = None

        if not os.path.exists(os.path.join(get_states_path(), Config.Paths.MAP)):
            print(f"Map directory not found at {os.path.join(get_states_path(), Config.Paths.MAP)}")
            self.random_states = []
        else:
            self.random_states = [os.path.join(Config.Paths.MAP, state) for state in os.listdir(os.path.join(get_states_path(), Config.Paths.MAP)) if state.endswith(".bin")]

        self.model.train()

    def __str__(self) -> str:
        return (f"x position: {self.current_state[0].item():<8.2f} y position: {self.current_state[1].item():<8.2f} next turn: {self.current_state[2].item():<8} "
                f"velocity: {self.current_state[3].item():<8.2f} acceleration: {self.current_state[4].item():<8.2f} relative yaw: {self.current_state[5].item():<8.2f} "
                f"next edge length: {self.current_state[6].item():<8.2f} second turn: {self.current_state[7].item():<8} third edge length: {self.current_state[8].item():<8.2f} "
                f"third turn: {self.current_state[9].item():<8}")

    def load_model(self) -> None:
        if self.model_path.qsize() > 0:
            path = self.model_path.get()
            model_pth = os.path.join(path, Config.Paths.MODEL_FILE_NAME)
            if os.path.exists(model_pth):
                self.hyperparameters = self.load_hyperparameters(path)
                self.model.load_state_dict(torch.load(model_pth, map_location=self.device))
                self.trainer = QTrainer(self.model, self.device, self.hyperparameters["learning_rate"], self.hyperparameters["gamma"])
                self.n_step_buffer = NStepBuffer(self.hyperparameters["n_steps"], self.device)
                self.memory = PrioritizedReplayBuffer(self.hyperparameters["max_memory"], alpha=self.hyperparameters["alpha"], beta=self.hyperparameters["beta_start"]) 
                self.logger.load(os.path.join(path, Config.Paths.STAT_FILE_NAME))
                print(f"Model loaded from {model_pth}")
            else:
                print(f"Model not found at {model_pth}")
        else:
            # Load fresh model with random weights
            self.hyperparameters = Config.NN.get_hyperparameters()
            self.model = Model().to(self.device)
            self.trainer = QTrainer(self.model, self.device, self.hyperparameters["learning_rate"], self.hyperparameters["gamma"])
            self.n_step_buffer = NStepBuffer(self.hyperparameters["n_steps"], self.device)
            self.memory = PrioritizedReplayBuffer(self.hyperparameters["max_memory"], alpha=self.hyperparameters["alpha"], beta=self.hyperparameters["beta_start"]) 
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

    
    def update_state(self, simulation_state: SimStateData) -> None:

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

        self.current_state = torch.tensor([
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

    
    def update_epsilon(self):
        if self.epsilon_dict["manual"]:
            self.epsilon = self.epsilon_dict["value"]
        else:
            self.epsilon = self.hyperparameters["epsilon_end"] + \
                (self.hyperparameters["epsilon_start"] - self.hyperparameters["epsilon_end"]) \
                      * np.exp(-1. * self.iterations / self.hyperparameters["epsilon_decay"])

    
    def get_action(self, state) -> torch.Tensor:
        self.update_epsilon()

        if random.random() < self.epsilon:
            move = torch.randint(0, Config.NN.Arch.OUTPUT_SIZE, (), device=self.device)
            # for i, action in enumerate(Config.NN.Arch.OUTPUTS_DESC):
            #     self.q_values_dict[action] = 0.0
            # self.q_values_dict[Config.NN.Arch.OUTPUTS_DESC[move]] = 1.0
            # self.q_values_dict["is_random"] = True
            return move
        else:
            with torch.no_grad():
                prediction = self.model(state)
                # for i, action in enumerate(Config.NN.Arch.OUTPUTS_DESC):
                #     self.q_values_dict[action] = prediction[i].item()
                # self.q_values_dict["is_random"] = False
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

    def get_reward(self, simulation_state: SimStateData) -> torch.Tensor:
        if self.prev_position is None:
            return torch.tensor(0, device=self.device)
        prev_position = self.prev_position
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

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done)) # No priority, it will take the highest priority by default

    def train_long_memory(self):
        batch = self.memory.sample(self.hyperparameters["batch_size"])
        if batch is None:
            return
        else:
            sample, indices, weights = batch
        states, actions, rewards, next_states, dones = zip(*sample)
        states =  torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.stack(dones).to(self.device)

        td_sample = self.trainer.train_step(states, actions, rewards, next_states, dones, weights)
        self.memory.update_priorities(indices, td_sample)

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        if _time == 20:
            if Config.Game.RANDOM_SPAWN:
                iface.execute_command(f"load_state {random.choice(self.random_states)}")
            self.ready = True
            if self.epsilon == 0.0:
                random_steering_value = np.random.randint(-10000, 10000)
                iface.execute_command(f"steer {random_steering_value}; press up")
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
            self.n_step_buffer.add(self.current_state, action, current_reward)        
            self.prev_position = simulation_state.position[0], simulation_state.position[2]  # Get the current position
            self.prev_positions.append(self.prev_position)

            self.send_input(iface, action)                # Send the action to the game

            if self.n_step_buffer.is_full() and not self.epsilon_dict["manual"]:
                state, action, reward = self.n_step_buffer.get_transition()
                next_state = self.current_state
                self.remember(state, action, reward, next_state, done)

            end_time = time.time()
            total_time = end_time - start_time
            # if total_time * 1000 > Config.Game.INTERVAL_BETWEEN_ACTIONS / Config.Game.GAME_SPEED:
            #     print(f"Warning: the action took {total_time * 1000:.2f}ms to execute, it should've taken less than {Config.Game.INTERVAL_BETWEEN_ACTIONS / Config.Game.GAME_SPEED:.2f}ms")

            if done:
                self.ready = False
                if not self.epsilon_dict["manual"]:
                    while not self.n_step_buffer.is_empty():
                        state, action, reward = self.n_step_buffer.get_transition()
                        next_state = self.current_state
                        self.n_step_buffer.pop_transition()
                        self.remember(state, action, reward, next_state, done)

                    self.iterations += 1
                    self.rewards_queue.put(self.reward)
                    self.logger.add_run(self.iterations, _time, self.reward)
                    self.train_long_memory()
                    if self.iterations % Config.NN.UPDATE_TARGET_EVERY == 0:
                        self.trainer.update_target()
                print(f"Iteration: {self.iterations:<8} reward: {self.reward:<8.2f} epsilon: {self.epsilon:<8.3f}")
                self.reward = 0.0
                self.prev_position = None
                self.prev_positions.clear()
                self.n_step_buffer.clear()
                self.prev_velocity = None
                self.epsilon_dict["value"] = self.epsilon
                if self.has_finished:
                    self.has_finished = False
                    self.launch_map(iface)
                else:
                    iface.horn()
                    iface.execute_command(f"load_state {self.random_states[0]}")    # State 0 of HorizonUnlimited

class NStepBuffer:
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

    
    def get_transition(self):
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

    def add(self, state, action, reward):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward

        self.position = (self.position + 1) % self.n_steps
        if self.current_size < self.n_steps:
            self.current_size += 1


    

