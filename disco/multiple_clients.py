import subprocess
import keyboard
from utils import *
from worker import Worker
import multiprocessing

num_clients = 6
choose_map_event = multiprocessing.Event()

def launch_games():
    for i in range(num_clients):
        launch_game()
        sleep(2)

def launch_game():
    executable, path_to_executable = get_executable_path()
    subprocess.Popen([executable], cwd=path_to_executable, shell=True)

def trigger_map_event():
    choose_map_event.set()
    sleep(2)
    focus_windows_by_name("TrackMania Nations Forever (TMInterface 1.4.3)")

if __name__ == "__main__":
    launch_games()
    pids = get_process_pids_by_name("TMForever.exe")
    move_windows(pids)

    servers = [i for i in range(num_clients)]

    # Create processes
    workers = []
    for server in servers:
        worker = Worker(server, choose_map_event)
        workers.append(worker)
        worker.start()

    print("Workers started")
    print("Press 'm' to choose map")
    print("Press 'f' to focus windows")
    print("Press 'CTRL+C' to quit")

    keyboard.add_hotkey('m', lambda: trigger_map_event())
    keyboard.add_hotkey('f', lambda: focus_windows_by_name("TrackMania Nations Forever (TMInterface 1.4.3)"))

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
