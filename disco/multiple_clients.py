from tminterface.client import Client, run_client
from tminterface.interface import TMInterface
import sys
import multiprocessing as mp
import numpy as np

class MainClient(Client):
    def __init__(self, num) -> None:
        super(MainClient, self).__init__()
        self.prev_state = None
        self.num = num

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time >= 0:
            state = iface.get_simulation_state()
            if self.num == 0:
                iface.execute_command(f"press up; steer {np.sin(_time / 1000) * 65535}")
            else:
                iface.execute_command(f"press down; steer {np.sin(_time / 1000) * 65535}")
            self.prev_state = state

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        self.log(iface, f"Checkpoint {current}/{target}")

    def log(self, iface, msg):
        iface.execute_command(f"log {msg}")


def worker(server_num):
    server_name = f"TMInterface{server_num}"
    print(f"Connecting to {server_name}...")
    client = MainClient(server_num)
    run_client(client, server_name)


if __name__ == '__main__':
    num_clients = 2
    servers = [i for i in range(num_clients)]
    with mp.Pool(num_clients) as pool:
        pool.map(worker, servers)
