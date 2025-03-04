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

def get_hwnds_for_pid(pid):
    def callback(hwnd, hwnds):
        #if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
        _, found_pid = win32process.GetWindowThreadProcessId(hwnd)

        if found_pid == pid:
            hwnds.append(hwnd)
        return True
    hwnds = []      # list to store handles of windows
    win32gui.EnumWindows(callback, hwnds)
    return hwnds

def focus_window(hwnd):
    if hwnd:
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys('%')
        win32gui.SetForegroundWindow(hwnd)
        sleep(0.1)
        rect = win32gui.GetWindowRect(hwnd)
        x = rect[0] + 50
        y = rect[1] + 50
        pyautogui.click(x, y)
    else:
        print("Window not found")

def focus_window_by_pid(pid):
    hwnds = get_hwnds_for_pid(pid)
    for hwnd in hwnds:
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
        x = (i % windows_horizontally) * w
        y = (i // windows_horizontally % windows_vertically) * h

        hwnds = get_hwnds_for_pid(pid)
        for hwnd in hwnds:
            win32gui.MoveWindow(hwnd, x, y, w, h, True)

