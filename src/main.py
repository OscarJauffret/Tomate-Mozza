import multiprocessing
from time import sleep

from .config import Config
from .utils.utils import trigger_map_event
from .horizon.worker import Worker
from .utils.tm_launcher import TMLauncher
from .utils.hotkey_manager import HotkeyManager
from .utils.plot import Plot

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

    queue = multiprocessing.Queue()
    plot = Plot(plot_size=20000, title="Reward", xlabel="Iteration", ylabel="Reward")

    # Create processes
    workers = []
    for server in servers:
        worker = Worker(server, choose_map_event, print_state_event, save_model_event, queue)
        workers.append(worker)
        worker.start()

    # Add hotkeys
    hotkey_manager.add_hotkey('m', lambda: trigger_map_event(choose_map_event), "load the map")
    hotkey_manager.add_hotkey('p', lambda: print_state_event.set(), "print the state")
    hotkey_manager.add_hotkey('f', lambda: launcher.focus_windows(), "focus the windows")
    hotkey_manager.add_hotkey('s', lambda: save_model_event.set(), "save the model")
    hotkey_manager.add_hotkey('f7', lambda: hotkey_manager.toggle_hotkeys(), "toggle hotkeys")

    hotkey_manager.print_hotkeys()


    # Main loop
    try:
        while all(worker.is_alive() for worker in workers):
            if queue.empty():
                sleep(0.1)
            else:
                reward = queue.get()
                plot.add_point(reward)
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        print("Workers terminated")
