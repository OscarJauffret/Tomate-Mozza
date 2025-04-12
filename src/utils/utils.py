import os
import torch
import time
import functools
import shutil

from dotenv import load_dotenv
from time import sleep
from tminterface.interface import TMInterface

from .tm_launcher import TMLauncher
from ..config import Config

load_dotenv()
profile_times = {}

def get_executable_path() -> tuple[str, str]:
    return os.getenv("EXECUTABLE"), os.getenv("EXECUTABLE_PATH")

def get_default_map() -> str:
    return os.path.join(os.getenv("DEFAULT_MAP_PATH"), os.getenv("DEFAULT_MAP_NAME"))

def get_states_path() -> str:
    return os.getenv("STATES_PATH")

def get_random_states() -> list[str]:
    """
    Get the random states from the TMInterface directory
    :return: the random states
    """
    if not os.path.exists(os.path.join(get_states_path(), Config.Paths.MAP)):
        print(f"Map directory not found at {os.path.join(get_states_path(), Config.Paths.MAP)}")
        return []
    else:
        return [os.path.join(Config.Paths.MAP, state) for state in
                os.listdir(os.path.join(get_states_path(), Config.Paths.MAP))
                if state.endswith(".bin")]

def trigger_map_event(event):
    event.set()
    sleep(2)
    TMLauncher.remove_fps_cap()
    TMLauncher.focus_windows()

def copy_model_to_latest(directory: str) -> None:
    """
    Copy the model to the latest directory
    :return: None
    """
    for file in os.listdir(directory):
        shutil.copy(os.path.join(directory, file), Config.Paths.LATEST_MODEL_PATH)


def get_device_info(device: str):
    """
    Get the device information
    :param device: The device to get the information for 'cuda' or 'cpu'
    :return: The device name
    """
    device_name = "CPU"
    if device == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        import platform
        if platform.system() == "Windows":
            import subprocess
            result = subprocess.check_output("wmic cpu get name", shell=True).decode().strip().split("\n")[1]
            device_name = result
        else:
            import os
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            device_name = line.split(':')[1].strip()
                            break
    return device_name

def profile_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if func.__name__ not in profile_times:
            profile_times[func.__name__] = []
        profile_times[func.__name__].append(end_time - start_time)
        return result
    return wrapper

def print_profile_times():
    for func_name, times in profile_times.items():
        avg_time = sum(times) / len(times)
        print(f"{func_name}: {avg_time * 1000:.4f} ms (called {len(times)} times)")