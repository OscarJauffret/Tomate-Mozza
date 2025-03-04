import subprocess
from utils import *
from worker import Worker

num_clients = 4

def launch_games():
    for i in range(num_clients):
        launch_game()
        sleep(2)

def launch_game():
    executable, path_to_executable = get_executable_path()
    subprocess.Popen([executable], cwd=path_to_executable, shell=True)

if __name__ == "__main__":
    launch_games()
    pids = get_process_pids_by_name("TMForever.exe")
    move_windows(pids)

    servers = [i for i in range(num_clients)]

    # Create processes
    workers = []
    for server in servers:
        worker = Worker(server)
        workers.append(worker)
        worker.start()

    # sleep(3)
    # focus_windows_by_pids(pids)
    # print("Windows focused")

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
