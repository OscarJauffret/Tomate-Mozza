from numpy.random import get_state
from tminterface.client import Client
from tminterface.interface import TMInterface
from utils import *
from model import Model, QTrainer
import config
import torch
import random

class HorizonClient(Client):
    def __init__(self, num) -> None:
        super(HorizonClient, self).__init__()
        self.prev_state = None
        self.num = num
        self.model = Model()
        self.trainer = QTrainer(self.model, config.LEARNING_RATE, config.GAMMA)

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

        current_state = [
            state.position[0] / 1024,
            state.position[2] / 1024,
            next_turn[0][0] / 32,
            next_turn[0][1] / 32,
            next_turn[1],
            state.yaw_pitch_roll[0]
        ]

        return current_state

    def get_action(self, state):
        epsilon = 0.9
        if np.random.random() > epsilon:
            final_move = np.random.random(3)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            final_move = self.model(state0)

        return final_move

    def send_input(self, iface: TMInterface, move) -> None:
        command = ""
        if move[0] > 0.8:
            command += "press up;"
        if move[1] > move[2]:
            command += "press left;"
        elif move[1] < move[2]:
            command += "press right;"
        iface.execute_command(command)

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        if _time >= 0:
            state = iface.get_simulation_state()

            state_old = self.get_state(iface)
            action = self.get_action(state_old)
            self.send_input(iface, action)


            #if self.num == 0:
            #    iface.execute_command(f"press up")
            #else:
            #    iface.execute_command(f"press down")
            self.prev_state = state

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        self.log(iface, f"Checkpoint {current}/{target}")

    def log(self, iface, msg):
        iface.execute_command(f"log {msg}")