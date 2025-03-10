import subprocess
from math import floor

from dotenv import load_dotenv
import os
from time import sleep
import pygetwindow as gw
import pywinauto
import config
import ast
import numpy as np


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

def get_current_block(pos_x, pos_y):
    block_x = floor(pos_x / 32)
    block_y = floor(pos_y / 32)
    return block_x, block_y

def load_map_layout():
    with open(config.MAP_LAYOUT_PATH, "r") as f:
        ordered_blocks = f.read()
        ordered_blocks = np.array(ast.literal_eval(ordered_blocks))
    return ordered_blocks

def get_block_index(block_x, block_y):
    for i, block in enumerate(ordered_blocks):
        if block[0] == block_x and block[1] == block_y:
            return i
    return -1

def get_next_turn(current_block_index):
    current_x, current_y = ordered_blocks[current_block_index]
    for i in range(current_block_index, len(ordered_blocks)):
        if ordered_blocks[i][0] != current_x and ordered_blocks[i][1] != current_y:
            direction = ordered_blocks[i - 1] - ordered_blocks[i - 2]
            if np.array_equal(direction, (0, 1)):     # Up
                if ordered_blocks[i][0] > current_x: # Right turn
                    direction = -1.0
                else: # Left turn
                    direction = 1.0
            elif np.array_equal(direction, (0, -1)):  # Down
                if ordered_blocks[i][0] > current_x: # Left turn
                    direction = 1.0
                else: # Right turn
                    direction = -1.0
            elif np.array_equal(direction, (1, 0)):  # Right
                if ordered_blocks[i][1] > current_y:
                    direction = 1.0
                else:
                    direction = -1.0
            elif np.array_equal(direction, (-1, 0)):  # Left
                if ordered_blocks[i][1] > current_y:
                    direction = -1.0
                else:
                    direction = 1.0

            return ordered_blocks[i - 1], direction
    return -1


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

ordered_blocks = load_map_layout()