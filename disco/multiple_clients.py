import time
from time import sleep
from tminterface.client import Client, run_client
from tminterface.interface import TMInterface
import multiprocessing as mp
import numpy as np
import subprocess
from utils import *
import pyautogui


class MainClient(Client):
    def __init__(self, num) -> None:
        super(MainClient, self).__init__()
        self.prev_state = None
        self.num = num

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")
        sleep(10)
        iface.execute_command("toggle_console")
        pyautogui.press('tab')
        sleep(1)
        pyautogui.press('enter')
        pyautogui.press('execute')
        pyautogui.press('accept')
        pyautogui.press('esc')
        sleep(1)
        iface.execute_command(f"map {get_default_map()}")

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
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


def launch_game():
    executable, path_to_executable = get_executable_path()
    result = subprocess.run([executable, "-PassThru"], cwd=path_to_executable, shell=True, text=True)
    print(result.stdout)

def worker(server_num):
    sleep(server_num*2)
    launch_game()
    server_name = f"TMInterface{server_num}"
    print(f"Connecting to {server_name}...")
    client = MainClient(server_num)
    run_client(client, server_name)


if __name__ == "__main__":
    num_clients = 1
    servers = [i for i in range(num_clients)]
    with mp.Pool(num_clients) as pool:
        pool.map(worker, servers)
