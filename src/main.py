import keyboard
from utils import *
from worker import Worker
import multiprocessing
import config

choose_map_event = multiprocessing.Event()
print_state_event = multiprocessing.Event()

if __name__ == "__main__":
    launch_games(config.NUMBER_OF_CLIENTS)
    sleep(1)
    move_windows_by_name("TrackMania Nations Forever (TMInterface 1.4.3)")

    servers = [i for i in range(config.NUMBER_OF_CLIENTS)]

    # Create processes
    workers = []
    for server in servers:
        worker = Worker(server, choose_map_event, print_state_event)
        workers.append(worker)
        worker.start()

    print("Workers started")
    print("Press 'm' to choose map")
    print("Press 'f' to focus windows")
    print("Press 'CTRL+C' to quit")

    keyboard.add_hotkey('m', lambda: trigger_map_event(choose_map_event))
    keyboard.add_hotkey('p', lambda: print_state_event.set())
    keyboard.add_hotkey('f', lambda: focus_windows_by_name(config.WINDOW_NAME))

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
