import keyboard
from utils import trigger_map_event
from worker import Worker
import multiprocessing
from config import Config
from tm_launcher import TMLauncher
from time import sleep
from hotkey_manager import HotkeyManager

choose_map_event = multiprocessing.Event()
print_state_event = multiprocessing.Event()
save_model_event = multiprocessing.Event()

if __name__ == "__main__":
    launcher = TMLauncher(Config.Game.NUMBER_OF_CLIENTS)
    launcher.launch_games()
    sleep(1)
    launcher.focus_windows()

    servers = [i for i in range(Config.Game.NUMBER_OF_CLIENTS)]
    hotkey_manager = HotkeyManager()


    # Create processes
    workers = []
    for server in servers:
        worker = Worker(server, choose_map_event, print_state_event, save_model_event)
        workers.append(worker)
        worker.start()

    # Add hotkeys
    hotkey_manager.add_hotkey('m', lambda: trigger_map_event(choose_map_event), "load the map")
    hotkey_manager.add_hotkey('p', lambda: print_state_event.set(), "print the state")
    hotkey_manager.add_hotkey('f', lambda: launcher.focus_windows(), "focus the windows")
    hotkey_manager.add_hotkey('s', lambda: save_model_event.set(), "save the model")
    hotkey_manager.add_hotkey('f7', lambda: hotkey_manager.toggle_hotkeys(), "toggle hotkeys")

    hotkey_manager.print_hotkeys()

    # Wait for all processes to finish
    try:
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        print("Workers terminated")
