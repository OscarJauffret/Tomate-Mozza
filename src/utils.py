import subprocess

from dotenv import load_dotenv
import os
from time import sleep
import pygetwindow as gw
import pywinauto
import config

load_dotenv()

def get_executable_path() -> tuple[str, str]:
    return os.getenv("EXECUTABLE"), os.getenv("EXECUTABLE_PATH")


def get_default_map() -> str:
    return os.path.join(os.getenv("DEFAULT_MAP_PATH"), os.getenv("DEFAULT_MAP_NAME"))


def launch_games(number_of_clients):
    for i in range(number_of_clients):
        launch_game()
        sleep(2)

def launch_game():
    executable, path_to_executable = get_executable_path()
    subprocess.Popen([executable], cwd=path_to_executable, shell=True)


def trigger_map_event(event):
    event.set()
    sleep(2)
    focus_windows_by_name(config.WINDOW_NAME)


def focus_windows_by_name(name):
    windows = gw.getWindowsWithTitle(name)
    print(f"Focusing {len(windows)} windows")
    for window in windows:
        app = pywinauto.Application().connect(handle=window._hWnd)
        dlg = app.top_window()
        dlg.set_focus()
        sleep(0.3)


def move_windows_by_name(name):
    w, h = 640, 480
    windows_horizontally = 1920 // w
    windows_vertically = 1080 // h

    windows = gw.getWindowsWithTitle(name)
    print(f"Moving {len(windows)} windows")
    for i, window in enumerate(windows):
        x = (i % windows_horizontally) * w
        y = (i // windows_horizontally % windows_vertically) * h
        window.moveTo(x, y)
