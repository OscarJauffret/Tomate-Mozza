import subprocess
import pygetwindow as gw
import pywinauto
import psutil

from time import sleep
from ReadWriteMemory import ReadWriteMemory

from src.config import Config


class TMLauncher:
    def __init__(self, number_of_clients: int) -> None:
        self.number_of_clients = number_of_clients

    def launch_games(self) -> None:
        """
        Launch the number of clients specified when creating the TMLauncher object
        :return: None
        """
        for i in range(self.number_of_clients):
            self._launch_game()
            sleep(2)

    @staticmethod
    def _launch_game() -> None:
        """
        Launch a single Trackmania client
        :return: None
        """
        from .utils import get_executable_path

        executable, path_to_executable = get_executable_path()
        subprocess.Popen([executable], cwd=path_to_executable, shell=True)

    @staticmethod
    def focus_windows() -> None:
        """
        Focus all Trackmania windows
        :return: None
        """
        windows = gw.getWindowsWithTitle(Config.Game.TMI_WINDOW_NAME)
        print(f"Focusing {len(windows)} windows")
        for window in windows:
            app = pywinauto.Application().connect(handle=window._hWnd)
            dlg = app.top_window()
            dlg.set_focus()
            sleep(0.3)

    @staticmethod
    def move_windows_by_name() -> None:
        """
        Move all Trackmania windows to a grid
        :return: None
        """
        w, h = 640, 480
        windows_horizontally = 1920 // w
        windows_vertically = 1080 // h

        windows = gw.getWindowsWithTitle(Config.Game.TMI_WINDOW_NAME)
        print(f"Moving {len(windows)} windows")
        for i, window in enumerate(windows):
            x = (i % windows_horizontally) * w
            y = (i // windows_horizontally % windows_vertically) * h
            window.moveTo(x, y)


    @staticmethod
    def remove_fps_cap():
        """
        Remove the FPS cap in Trackmania. It used reverse engineering to find the memory addresses of the FPS cap.
        :return: None
        """
        process = filter(lambda p: p.name() == Config.Game.PROCESS_NAME, psutil.process_iter())
        rwm = ReadWriteMemory()
        for p in process:
            pid = int(p.pid)
            process = rwm.get_process_by_id(pid)
            process.open()
            process.write(0x005292F1, 4294919657)
            process.write(0x005292F1 + 4, 2425393407)
            process.write(0x005292F1 + 8, 2425393296)
            process.close()
