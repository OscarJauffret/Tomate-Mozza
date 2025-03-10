from tminterface.client import Client
from tminterface.interface import TMInterface
from utils import *

class HorizonClient(Client):
    def __init__(self, num) -> None:
        super(HorizonClient, self).__init__()
        self.prev_state = None
        self.num = num

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

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        if _time >= 0:
            state = iface.get_simulation_state()

            #if self.num == 0:
            #    iface.execute_command(f"press up")
            #else:
            #    iface.execute_command(f"press down")
            self.prev_state = state

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        self.log(iface, f"Checkpoint {current}/{target}")

    def log(self, iface, msg):
        iface.execute_command(f"log {msg}")