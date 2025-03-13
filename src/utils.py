from dotenv import load_dotenv
import os
from time import sleep
from tm_launcher import TMLauncher

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
