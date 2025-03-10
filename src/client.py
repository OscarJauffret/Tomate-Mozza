from tminterface.client import Client
from tminterface.interface import TMInterface
from utils import *
from model import Model, QTrainer
import config
import torch
from collections import deque
import random

class HorizonClient(Client):
    def __init__(self, num) -> None:
        super(HorizonClient, self).__init__()
        self.prev_state = None
        self.num = num
        self.memory = deque(maxlen=config.MAX_MEMORY)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = Model().to(self.device)
        self.trainer = QTrainer(self.model, self.device, config.LEARNING_RATE, config.GAMMA)
        self.reward = 0
        self.iteration_done = False
        self.iterations = 0

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")

    def launch_map(self, iface: TMInterface) -> None:
        iface.execute_command(f"map {get_default_map()}")

    def print_state(self, iface: TMInterface) -> None:
        state = iface.get_simulation_state()
        print(f"Position: {state.position}")
        print(f"Block position: {get_current_block(state.position[0], state.position[2])}")
        print(f"Block index: {get_block_index(*get_current_block(state.position[0], state.position[2]))}")
        print(f"Next turn: {get_next_turn(get_block_index(*get_current_block(state.position[0], state.position[2])))}")
        print(f"Yaw, pitch roll: {state.yaw_pitch_roll}")
        print(f"Velocity: {state.velocity}")
        print(f"Race time: {state.race_time}")
        print(f"Input finish event: {state.input_finish_event}")

    def get_state(self, iface: TMInterface):
        state = iface.get_simulation_state()
        current_block = get_current_block(state.position[0], state.position[2])
        block_index = get_block_index(*current_block)
        next_turn = get_next_turn(block_index)

        current_state = torch.tensor([
            state.position[0] / 1024,
            state.position[2] / 1024,
            next_turn[0][0] / 32,
            next_turn[0][1] / 32,
            next_turn[1],
            state.yaw_pitch_roll[0]
        ], dtype=torch.float, device=self.device)

        return current_state

    def get_action(self, state):
        epsilon = 0.7
        move = [0] * config.OUTPUT_SIZE
        if np.random.random() > epsilon:
            final_move = np.random.randint(0, config.OUTPUT_SIZE)
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

    def get_reward(self, state):
        current_reward = get_current_block(state[0] * 1024, state[1] * 1024)
        current_reward = get_block_index(*current_reward)
        return torch.tensor(current_reward, device=self.device)

    def determine_done(self, iface: TMInterface, state_new):
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
        if len(self.memory) > config.BATCH_SIZE:
            mini_sample = random.sample(self.memory, config.BATCH_SIZE)        # List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states =  torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
             
        if _time >= 0 and not self.iteration_done and _time % 100 == 0:
            
            state_old = self.get_state(iface)
            action = self.get_action(state_old)
            self.send_input(iface, action)

            state_new = self.get_state(iface)
            current_reward = self.get_reward(state_new)
            self.iteration_done = self.determine_done(iface, state_new)

            self.train_short_memory(state_old, action, current_reward, state_new, self.iteration_done)
            self.remember(state_old, action, current_reward, state_new, self.iteration_done)

            self.reward += current_reward
            if self.iteration_done:
                self.iterations += 1
                print(f"Iteration: {self.iterations}")
                print(f"DONE: {self.reward}")
                self.train_long_memory()
                self.reward = 0
                iface.horn()
                iface.respawn()
                self.iteration_done = False
