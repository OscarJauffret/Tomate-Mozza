import multiprocessing
import os

from .config import Config
from .horizon.worker import Worker
from .app.interface import Interface
from .horizon.events import Events

from argparse import ArgumentParser

events = Events()

argparser = ArgumentParser(description="Horizon - A reinforcement learning framework for TM")
argparser.add_argument("--alg", type=str, default="ppo", help="Algorithm to use (ppo, dqn)")
argparser.add_argument("--name", type=str, default="", help="Path to save the model")


if __name__ == "__main__":
    args = argparser.parse_args()

    if args.name:
        args.name = os.path.join(Config.Paths.MODELS_PATH, args.name)

    manager = multiprocessing.Manager()
    outputs = Config.Arch.OUTPUTS_DESC
    shared_dict = manager.dict({
                                "eval": False,
                                "reward": manager.Queue(),
                                "q_values": manager.dict({outputs[0]: 0, outputs[1]: 0, outputs[2]: 0, outputs[3]: 0, outputs[4]: 0, outputs[5]: 0,
                                                          outputs[6]: 0, outputs[7]: 0, outputs[8]: 0, outputs[9]: 0, outputs[10]: 0, outputs[11]: 0,
                                                          "is_random": False}),
                                "model_path": manager.Value("u", args.name),
                                "game_speed": Config.Game.GAME_SPEED,
                                "personal_best": float("inf"),
                                })
    app = Interface(events, shared_dict)

    # Create processes
    worker = Worker(args.alg.upper(), events, shared_dict)
    worker.start()

    # Main loop
    try:
        app.update_interface()
        app.run()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join()
        print("Workers terminated")
