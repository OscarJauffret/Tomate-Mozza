import subprocess
from math import floor

from dotenv import load_dotenv
import os
from time import sleep
import pygetwindow as gw
import pywinauto
from config import Config
import ast
import numpy as np
import json

load_dotenv()

def get_executable_path() -> tuple[str, str]:
    return os.getenv("EXECUTABLE"), os.getenv("EXECUTABLE_PATH")


def get_default_map() -> str:
    return os.path.join(os.getenv("DEFAULT_MAP_PATH"), os.getenv("DEFAULT_MAP_NAME"))

def launch_games(number_of_clients):
    for i in range(number_of_clients):
        _launch_game()
        sleep(2)

def _launch_game():
    executable, path_to_executable = get_executable_path()
    subprocess.Popen([executable], cwd=path_to_executable, shell=True)

def get_section_info(pos_x, pos_y):
    """
    Get the section information that is passed in the state of the client. The info contains the relative position of the
    car in the section and the next turn.
    :param pos_x: The x position of the car (absolute)
    :param pos_y: The y position of the car (absolute)
    :return: A tuple containing the relative position of the car in the section and the next turn
    """
    pos_in_section, section_index = _get_position_relative_to_section(pos_x, pos_y)
    if pos_in_section == (-1, -1):
        return (-1, -1), 0
    next_turn = turns[section_index]
    return pos_in_section, next_turn

def _get_position_relative_to_section(pos_x, pos_y):
    block_x, block_y = get_current_block(pos_x, pos_y)
    for i, section in enumerate(sections):
        if _is_in_section(block_x, block_y, section):
            start_x, start_y = section[0]

            half_block_size = 16
            start_x = start_x * 32 + half_block_size     # 32 is the size of a block, 16 is the center of the block
            start_y = start_y * 32 + half_block_size     # This will be the origin of our coordinate system
            end_x, end_y = section[1]
            end_x = end_x * 32 + half_block_size
            end_y = end_y * 32 + half_block_size

            translated_x = pos_x - start_x
            translated_y = pos_y - start_y

            direction = (end_x - start_x, end_y - start_y)
            # Normalize the direction
            direction_norm = [0, 0]
            for i in range(2):
                direction_norm[i] = 0 if direction[i] == 0 else int(direction[i] / abs(direction[i]))

            angles = {(1, 0): 0, (0, 1): np.pi/2, (-1, 0): np.pi, (0, -1): 3*np.pi/2}
            direction_norm_tuple = tuple(direction_norm)
            angle = angles[direction_norm_tuple]

            rotated_x = translated_x * np.cos(-angle) - translated_y * np.sin(-angle)
            rotated_y = translated_x * np.sin(-angle) + translated_y * np.cos(-angle)

            length_of_section = np.linalg.norm(direction)
            size = length_of_section, half_block_size
            normalized_x = rotated_x / size[0]
            normalized_y = rotated_y / size[1]

            return (normalized_x, normalized_y), i

    return (-1, -1), -1

def _is_in_section(block_x, block_y, section):
    [start_x, start_y], [end_x, end_y] = section
    start_x, end_x = min(start_x, end_x), max(start_x, end_x)
    start_y, end_y = min(start_y, end_y), max(start_y, end_y)
    return start_x <= block_x <= end_x and start_y <= block_y <= end_y

def get_current_block(pos_x, pos_y):
    block_x = floor(pos_x / 32)
    block_y = floor(pos_y / 32)
    return block_x, block_y

def _get_section_dimension(section_index):
    start_x, start_y = sections[section_index][0]
    end_x, end_y = sections[section_index][1]

    return np.array([(abs(start_x - end_x) + abs(start_y - end_y)) * 32, 16])
    
def get_distance_reward(prev_pos, current_pos):
    prev_pos_in_section, prev_section_index = _get_position_relative_to_section(*prev_pos)
    current_pos_in_section, current_section_index = _get_position_relative_to_section(*current_pos)

    prev_section_dimension = _get_section_dimension(prev_section_index)
    current_section_dimension = _get_section_dimension(current_section_index)

    if prev_section_index == -1 or current_section_index == -1:
        return 0
    
    if prev_section_index == current_section_index:
        mul = -1 if current_pos_in_section[0] < prev_pos_in_section[0] else 1
        dist = np.linalg.norm(np.array(prev_pos_in_section) * prev_section_dimension - np.array(current_pos_in_section) * current_section_dimension)
        return mul * dist

    mul = -1 if current_section_index < prev_section_index else 1
    dist = np.linalg.norm(np.array(prev_pos) - np.array(current_pos))
    return mul * dist

def _load_map_layout():
    with open(Config.Paths.MAP_LAYOUT_PATH, "r") as f:
        data = json.load(f)
        blocks = np.array(data["layout"])
        sections = np.array(data["sections"])
        turns = data["turns"]
    return blocks, sections, turns

def get_block_index(block_x, block_y):
    for i, block in enumerate(blocks):
        if block[0] == block_x and block[1] == block_y:
            return i
    return -1

def trigger_map_event(event):
    event.set()
    sleep(2)
    focus_windows_by_name(Config.Game.WINDOW_NAME)

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

blocks, sections, turns = _load_map_layout()