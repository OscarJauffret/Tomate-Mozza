import os
import time
from time import sleep
from tminterface.client import Client, run_client
from tminterface.interface import TMInterface
import multiprocessing as mp
import numpy as np
import subprocess
from utils import *
import threading
import pywinctl
import psutil

registered_clients = 0
num_clients = 2
lock = threading.Lock()

class MainClient(Client):
    def __init__(self, num) -> None:
        super(MainClient, self).__init__()
        self.prev_state = None
        self.num = num

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")

        timer = threading.Timer(5, self.launch_map, args=(iface,))
        timer.start()

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


def launch_games():
    for i in range(num_clients):
        launch_game()
        sleep(2)

def launch_game():
    executable, path_to_executable = get_executable_path()
    subprocess.Popen([executable], cwd=path_to_executable, shell=True)

def worker(server_num):
    sleep(2*server_num)
    server_name = f"TMInterface{server_num}"
    print(f"Connecting to {server_name}...")
    client = MainClient(server_num)
    run_client(client, server_name)

if __name__ == "__main__":
    launch_games()
    pids = get_process_pids_by_name("TmForever.exe")
    move_windows(pids)

    servers = [i for i in range(num_clients)]

    # Create processes
    processes = []
    for server in servers:
        process = mp.Process(target=worker, args=(server,))
        processes.append(process)
        process.start()

    sleep(3)
    focus_windows_by_pids(pids)
    print("Windows focused")

    # Wait for all processes to finish
    for process in processes:
        process.join()
