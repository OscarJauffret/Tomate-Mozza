import os
import torch
import time
import functools

from dotenv import load_dotenv
from time import sleep

from .tm_launcher import TMLauncher

load_dotenv()

def get_executable_path() -> tuple[str, str]:
    return os.getenv("EXECUTABLE"), os.getenv("EXECUTABLE_PATH")

def get_default_map() -> str:
    return os.path.join(os.getenv("DEFAULT_MAP_PATH"), os.getenv("DEFAULT_MAP_NAME"))

def trigger_map_event(event):
    event.set()
    sleep(2)
    TMLauncher.remove_fps_cap()
    TMLauncher.focus_windows()

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
        print(f"{func.__name__} took {(end_time - start_time) * 1000:.2f}ms to execute")
        return result
    return wrapper