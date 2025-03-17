import torch
import random
import numpy as np

from tminterface.client import Client
from tminterface.interface import TMInterface
from collections import deque

from ..map_interaction.map_layout import MapLayout
from ..utils.utils import *
from .model import Model, QTrainer
from ..config import Config
from ..utils.tm_logger import TMLogger

class HorizonClient(Client):
    def __init__(self, num, queue) -> None:
        super(HorizonClient, self).__init__()
        self.num = num
        self.map_layout = MapLayout()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = Model().to(self.device)
        self.trainer = QTrainer(self.model, self.device, Config.NN.LEARNING_RATE, Config.NN.GAMMA)

        self.memory = deque(maxlen=Config.NN.MAX_MEMORY)
        self.reward = 0.0
        self.epsilon = Config.NN.EPSILON_START
        self.prev_position = None
        self.state = None
        self.iterations = 0
        self.ready = False

        self.logger = TMLogger(get_device_info(self.device.type))
        self.queue = queue

    def __str__(self) -> str:
        return f"x position: {self.state[0].item():<8.2f} y position: {self.state[1].item():<8.2f} next turn: {self.state[2].item():<8} yaw: {self.state[3].item():<8.2f}"

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")

    def launch_map(self, iface: TMInterface) -> None:
        iface.execute_command(f"map {get_default_map()}")

    def save_model(self) -> None:
        self.logger.update_log_id()
        directory = self.logger.dump()

        model_path = os.path.join(directory, "model.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def get_state(self, iface: TMInterface):
        state = iface.get_simulation_state()

        section_rel_pos, next_turn = self.map_layout.get_section_info(state.position[0], state.position[2])
        relative_yaw = self.map_layout.get_car_orientation(state.yaw_pitch_roll[0], state.position[0], state.position[2])

        current_state = torch.tensor([
            section_rel_pos[0],
            section_rel_pos[1],
            next_turn,
            relative_yaw
        ], dtype=torch.float, device=self.device)

        return current_state

    def get_action(self, state):
        self.epsilon = Config.NN.EPSILON_END + (Config.NN.EPSILON_START - Config.NN.EPSILON_END) * np.exp(-1. * self.iterations / Config.NN.EPSILON_DECAY)
        move = [0] * Config.NN.Arch.OUTPUT_SIZE
        if np.random.random() > self.epsilon:
            final_move = np.random.randint(0, Config.NN.Arch.OUTPUT_SIZE)
            move[final_move] = 1
        else:
            state0 = state.clone().detach().to(self.device)
            final_move = self.model(state0)
            final_move = torch.argmax(final_move).item()
            move[final_move] = 1

        return torch.tensor(move, device=self.device)

    def send_input(self, iface: TMInterface, move) -> None:
        if move[1] == 1:
            iface.set_input_state(accelerate=True, left=False, right=False)
        elif move[2] == 1:
            iface.set_input_state(accelerate=False, left=False, right=True)
        elif move[3] == 1:
            iface.set_input_state(accelerate=False, left=True, right=False)
        elif move[4] == 1:
            iface.set_input_state(accelerate=True, left=False, right=True)
        elif move[5] == 1:
            iface.set_input_state(accelerate=True, left=True, right=False)
        else: # Do nothing
            iface.set_input_state(accelerate=False, left=False, right=False)

    def get_reward(self, iface: TMInterface):
        if self.prev_position is None:
            return torch.tensor(0, device=self.device)
        current_position = iface.get_simulation_state().position[0], iface.get_simulation_state().position[2]
        current_reward = self.map_layout.get_distance_reward(self.prev_position, current_position)
        return torch.tensor(current_reward, device=self.device)

    def determine_done(self, iface: TMInterface):
        state = iface.get_simulation_state()

        if state.position[1] < 23: # If the car is below the track
            return True
        if not state.player_info.finish_not_passed:
            return True
        if state.display_speed < 5 and state.race_time > 2000:
            return True
        return False

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > Config.NN.BATCH_SIZE:
            mini_sample = random.sample(self.memory, Config.NN.BATCH_SIZE)        # List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states =  torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        if _time == 0:
            self.ready = True

        if _time >= 0 and _time % 100 == 0 and self.ready:
            state_old = self.get_state(iface)
            action = self.get_action(state_old)
            self.send_input(iface, action)

            state_new = self.get_state(iface)
            self.state = state_new
            current_reward = self.get_reward(iface)
            done = self.determine_done(iface)

            self.train_short_memory(state_old, action, current_reward, state_new, done)
            self.remember(state_old, action, current_reward, state_new, done)

            self.reward += current_reward
            self.prev_position = iface.get_simulation_state().position[0], iface.get_simulation_state().position[2]
            if done:
                self.ready = False

                self.iterations += 1
                print(f"Iteration: {self.iterations}, reward: {self.reward:.2f}, epsilon: {self.epsilon:.2f}")
                self.queue.put(self.reward.item())
                self.logger.add_run(self.iterations, _time, self.reward.item())

                self.train_long_memory()
                self.reward = 0.0
                self.prev_position = None
                iface.horn()
                iface.respawn()