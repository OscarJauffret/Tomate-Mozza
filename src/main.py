import multiprocessing
from time import sleep
import os

from .config import Config
from .utils.utils import trigger_map_event
from .horizon.worker import Worker
from .utils.tm_launcher import TMLauncher
from .utils.plot import Plot
from .app.interface import Interface

choose_map_event = multiprocessing.Event()
print_state_event = multiprocessing.Event()
save_model_event = multiprocessing.Event()
quit_event = multiprocessing.Event()

model_path = os.path.join(Config.Paths().LATEST_MODEL_PATH, "model.pth")
init_iterations = 0 #455       # TODO: Read from the log file

if __name__ == "__main__":
    launcher = TMLauncher(Config.Game.NUMBER_OF_CLIENTS)
    launcher.launch_games()
    sleep(1)
    launcher.focus_windows()

    servers = [i for i in range(Config.Game.NUMBER_OF_CLIENTS)]

    queue = multiprocessing.Queue()
    app = Interface(choose_map_event, print_state_event, save_model_event, quit_event)

    # Create processes
    workers = []
    for server in servers:
        worker = Worker(server, choose_map_event, print_state_event, save_model_event, quit_event, queue, model_path, init_iterations)
        workers.append(worker)
        worker.start()

    # Main loop
    try:
        app.update_graph(queue)
        app.run()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        print("Workers terminated")
