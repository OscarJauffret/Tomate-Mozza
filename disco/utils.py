import threading

import win32api
from dotenv import load_dotenv
import psutil
import pywinctl
import os
import win32gui, win32process, win32con, win32com.client
from time import sleep
import pyautogui

load_dotenv()

def get_executable_path() -> tuple[str, str]:
    return os.getenv("EXECUTABLE"), os.getenv("EXECUTABLE_PATH")

def get_default_map() -> str:
    return os.path.join(os.getenv("DEFAULT_MAP_PATH"), os.getenv("DEFAULT_MAP_NAME"))

def get_process_pids_by_name(process_name="TMForever.exe"):
    pids = []
    for process in psutil.process_iter(attrs=['pid', 'name']):
        if process.info['name'] and process_name.lower() in process.info['name'].lower():
            pids.append(process.info['pid'])

    return pids

def get_main_hwnd_for_pid(pid):
    def callback(hwnd, hwnds):
        _, found_pid = win32process.GetWindowThreadProcessId(hwnd)

        if found_pid == pid and win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            hwnds.append(hwnd)
        return True
    hwnds = []      # list to store handles of windows
    win32gui.EnumWindows(callback, hwnds)
    if hwnds:
        return hwnds[0]
    return None

def focus_window(hwnd):
    if hwnd:
        remote_thread, _ = win32process.GetWindowThreadProcessId(hwnd)
        win32process.AttachThreadInput(threading.get_ident(), remote_thread, True)
        win32gui.SetFocus(hwnd)
    else:
        print("Window not found")

def focus_window_by_pid(pid):
    hwnd = get_main_hwnd_for_pid(pid)
    focus_window(hwnd)

def focus_windows_by_pids(pids):
    for pid in pids:
        focus_window_by_pid(pid)
        sleep(3)

def move_windows(pids):
    w, h = 640, 480

    windows_horizontally = 1920 // w
    windows_vertically = 1080 // h
    for i, pid in enumerate(pids):
        hwnd = get_main_hwnd_for_pid(pid)
        if hwnd:
            x = (i % windows_horizontally) * w
            y = (i // windows_horizontally % windows_vertically) * h
            win32gui.MoveWindow(hwnd, x, y, w, h, True)
