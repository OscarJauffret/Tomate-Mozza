from tminterface.client import Client
from tminterface.interface import TMInterface
from utils import *

class MainClient(Client):
    def __init__(self, num) -> None:
        super(MainClient, self).__init__()
        self.prev_state = None
        self.num = num

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")

    def launch_map(self, iface: TMInterface) -> None:
        iface.execute_command(f"map {get_default_map()}")

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        if _time >= 0:
            state = iface.get_simulation_state()
            if self.num == 0:
                iface.execute_command(f"press up")
            else:
                iface.execute_command(f"press down")
            self.prev_state = state

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        self.log(iface, f"Checkpoint {current}/{target}")

    def log(self, iface, msg):
        iface.execute_command(f"log {msg}")