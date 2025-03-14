
class Config:
    DATETIME_FORMAT: str = '%m-%d_%H-%M'

    class Paths:
        MAP_LAYOUT_PATH: str = "maps/ordered_blocks.json"
        MAP_GBX_OUTPUT_PATH: str = "maps/horizon_layout.txt"
        MODELS_PATH: str = "models/"

    class Game:
        NUMBER_OF_CLIENTS: int = 1
        WINDOW_NAME: str = "TrackMania Nations Forever (TMInterface 1.4.3)"
        PROCESS_NAME: str = "TmForever.exe"

    class NN:
        INPUT_SIZE: int = 4
        OUTPUT_SIZE: int = 6
        LEARNING_RATE: float = 0.001
        GAMMA: float = 0.9
        MAX_MEMORY: int = 100_000
        BATCH_SIZE: int = 128
        EPSILON_START: float = 0.9
        EPSILON_END: float = 0.05
        EPSILON_DECAY: int = 10000

        @staticmethod
        def get_hyperparameters():
            return {
                "input_size": Config.NN.INPUT_SIZE,
                "output_size": Config.NN.OUTPUT_SIZE,
                "learning_rate": Config.NN.LEARNING_RATE,
                "gamma": Config.NN.GAMMA,
                "max_memory": Config.NN.MAX_MEMORY,
                "batch_size": Config.NN.BATCH_SIZE,
                "epsilon_start": Config.NN.EPSILON_START,
                "epsilon_end": Config.NN.EPSILON_END,
                "epsilon_decay": Config.NN.EPSILON_DECAY
            }