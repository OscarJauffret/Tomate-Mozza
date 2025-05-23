import multiprocessing
from time import sleep

from .config import Config
from .horizon.worker import Worker
from .utils.tm_launcher import TMLauncher
from .app.interface import Interface

from argparse import ArgumentParser

choose_map_event = multiprocessing.Event()
print_state_event = multiprocessing.Event()
load_model_event = multiprocessing.Event()
save_model_event = multiprocessing.Event()
quit_event = multiprocessing.Event()

argparser = ArgumentParser(description="Horizon - A reinforcement learning framework for TM")
argparser.add_argument("--alg", type=str, default="ppo", help="Algorithm to use (ppo, dqn)")


if __name__ == "__main__":
    args = argparser.parse_args()
    launcher = TMLauncher(Config.Game.NUMBER_OF_CLIENTS)
    launcher.launch_games()
    sleep(1)
    launcher.focus_windows()

    servers = [i for i in range(Config.Game.NUMBER_OF_CLIENTS)]

    manager = multiprocessing.Manager()
    outputs = Config.Arch.OUTPUTS_DESC
    shared_dict = manager.dict({
                                "eval": False,
                                "reward": manager.Queue(),
                                "q_values": manager.dict({outputs[0]: 0, outputs[1]: 0, outputs[2]: 0, outputs[3]: 0, outputs[4]: 0, outputs[5]: 0, "is_random": False}),
                                "model_path": manager.Value("u", ""),
                                "game_speed": Config.Game.GAME_SPEED,
                                "personal_best": float("inf"),
                                })
    app = Interface(choose_map_event, print_state_event, load_model_event, save_model_event, quit_event, shared_dict)

    # Create processes
    workers = []
    for server in servers:
        worker = Worker(server, args.alg.upper() ,choose_map_event, print_state_event, load_model_event,
                        save_model_event, quit_event, shared_dict)
        workers.append(worker)
        worker.start()

    # Main loop
    try:
        app.update_interface()
        app.run()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        print("Workers terminated")
