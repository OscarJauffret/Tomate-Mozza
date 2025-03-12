import keyboard
from utils import trigger_map_event
from worker import Worker
import multiprocessing
from config import Config
from tm_launcher import TMLauncher
from time import sleep

choose_map_event = multiprocessing.Event()
print_state_event = multiprocessing.Event()
save_model_event = multiprocessing.Event()

if __name__ == "__main__":
    launcher = TMLauncher(Config.Game.NUMBER_OF_CLIENTS)
    launcher.launch_games()
    sleep(1)
    launcher.focus_windows()

    servers = [i for i in range(Config.Game.NUMBER_OF_CLIENTS)]

    # Create processes
    workers = []
    for server in servers:
        worker = Worker(server, choose_map_event, print_state_event, save_model_event)
        workers.append(worker)
        worker.start()

    print("Workers started")
    print("Press 'm' to choose map")
    print("Press 'f' to focus windows")
    print("Press 'p' to print state")
    print("Press 's' to save the NN")
    print("Press 'CTRL+C' to quit")

    keyboard.add_hotkey('m', lambda: trigger_map_event(choose_map_event))
    keyboard.add_hotkey('p', lambda: print_state_event.set())
    keyboard.add_hotkey('f', lambda: launcher.focus_windows())
    keyboard.add_hotkey('s', lambda: save_model_event.set())

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
