from dotenv import load_dotenv
import psutil
import os
import win32gui, win32process
from time import sleep
import pygetwindow as gw
import pywinauto

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

def get_hwnds_of_windows_named(name):
    pids = get_process_pids_by_name(name)
    hwnds = []
    for pid in pids:
        hwnd = get_main_hwnd_for_pid(pid)
        if hwnd:
            hwnds.append(hwnd)
    return hwnds

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

def focus_windows_by_name(name):
    windows = gw.getWindowsWithTitle(name)
    print(f"Focusing {len(windows)} windows")
    for window in windows:
        app = pywinauto.Application().connect(handle=window._hWnd)
        dlg = app.top_window()
        dlg.set_focus()
        sleep(0.3)

def move_windows(pids):
    w, h = 640, 480

    windows_horizontally = 1920 // w
    windows_vertically = 1080 // h
    for i, pid in enumerate(pids):
        handle = get_main_hwnd_for_pid(pid)
        if handle:
            x = (i % windows_horizontally) * w
            y = (i // windows_horizontally % windows_vertically) * h

            win32gui.MoveWindow(handle, x, y, w, h, True)

