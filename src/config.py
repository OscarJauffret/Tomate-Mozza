
class Config:
    DATETIME_FORMAT: str = '%m-%d_%H-%M'

    class Paths:
        MAP_LAYOUT_PATH: str = "maps/ez_ordered_blocks.json"
        MAP_GBX_OUTPUT_PATH: str = "maps/Easy_pizzy.txt"
        MODELS_PATH: str = "models/"

    class Game:
        NUMBER_OF_CLIENTS: int = 1
        WINDOW_NAME: str = "TrackMania Nations Forever (TMInterface 1.4.3)"
        PROCESS_NAME: str = "TmForever.exe"

        BLOCK_SIZE: int = 32

        NUMBER_OF_ACTIONS_PER_SECOND: int = 10
        INTERVAL_BETWEEN_ACTIONS: int = 1000 // NUMBER_OF_ACTIONS_PER_SECOND
        GAME_SPEED: int = 12

    class NN:
        LEARNING_RATE: float = 0.005
        GAMMA: float = 0.9
        MAX_MEMORY: int = 100_000
        BATCH_SIZE: int = 128
        EPSILON_START: float = 0.9
        EPSILON_END: float = 0.05
        EPSILON_DECAY: int = 2000

        @staticmethod
        def get_hyperparameters():
            return {
                "learning_rate": Config.NN.LEARNING_RATE,
                "gamma": Config.NN.GAMMA,
                "max_memory": Config.NN.MAX_MEMORY,
                "batch_size": Config.NN.BATCH_SIZE,
                "epsilon_start": Config.NN.EPSILON_START,
                "epsilon_end": Config.NN.EPSILON_END,
                "epsilon_decay": Config.NN.EPSILON_DECAY
            }

        class Arch:
            INPUTS_DESC: list[str] = ["section_rel_x", "section_rel_y", "in_game_velocity", "relative_yaw"]
            OUTPUTS_DESC: list[str] = ["forward", "right", "left", "forward_right", "forward_left", "release"]
            REWARD_DESC: str = "total distance travelled multiplied by speed and a penalty factor"

            INPUT_SIZE: int = len(INPUTS_DESC)
            OUTPUT_SIZE: int = len(OUTPUTS_DESC)

            LAYER_SIZES: list[int] = [128, 128]
            NUMBER_OF_HIDDEN_LAYERS: int = len(LAYER_SIZES)

            @staticmethod
            def get_architecture_description():
                return {
                    "inputs": Config.NN.Arch.INPUTS_DESC,
                    "outputs": Config.NN.Arch.OUTPUTS_DESC,
                    "input_size": Config.NN.Arch.INPUT_SIZE,
                    "output_size": Config.NN.Arch.OUTPUT_SIZE,
                    "layer_sizes": Config.NN.Arch.LAYER_SIZES,
                    "number_of_hidden_layers": Config.NN.Arch.NUMBER_OF_HIDDEN_LAYERS,
                    "reward_description": Config.NN.Arch.REWARD_DESC
                }