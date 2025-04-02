import os

class Config:
    DATETIME_FORMAT: str = '%m-%d_%H-%M'

    class Paths:
        MAP_PREFIX: str = "maps"
        MAP: str = "horizon_unlimited"        # Verify that the map here is the same as the one in your .env file
        MAP_BLOCKS_PATH: str = os.path.join(MAP_PREFIX, MAP, "ordered_blocks.json")
        MAP_LAYOUT_PATH: str = os.path.join(MAP_PREFIX, MAP, "layout.txt")

        MODELS_PATH: str = "models/"
        LATEST_MODEL_PATH: str = os.path.join(MODELS_PATH, "latest")
        STAT_FILE_NAME: str = "stats.json"
        MODEL_FILE_NAME: str = "model.pth"

        @staticmethod
        def get_map():
            return {"ordered_blocks": Config.Paths.MAP_BLOCKS_PATH,
                    "list_of_blocks": Config.Paths.MAP_LAYOUT_PATH}

    class Game:
        NUMBER_OF_CLIENTS: int = 1
        WINDOW_NAME: str = "TrackMania Nations Forever (TMInterface 1.4.3)"
        PROCESS_NAME: str = "TmForever.exe"

        BLOCK_SIZE: int = 32

        NUMBER_OF_ACTIONS_PER_SECOND: int = 10
        INTERVAL_BETWEEN_ACTIONS: int = 1000 // NUMBER_OF_ACTIONS_PER_SECOND
        GAME_SPEED: int = 8

        RANDOM_SPAWN: bool = False

    class NN:
        LEARNING_RATE: float = 0.001
        GAMMA: float = 0.99
        MAX_MEMORY: int = 100_000
        MIN_MEMORY: int = 10_000
        BATCH_SIZE: int = 128
        EPSILON_START: float = 0.9
        EPSILON_END: float = 0.05
        EPSILON_DECAY: int = 10000
        UPDATE_TARGET_EVERY: int = 1000

        TAU: float = 0.005

        ALPHA: float = 0.7
        BETA_START: float = 0.4
        BETA_MAX: float = 1.0
        BETA_INCREMENT_STEPS: int = 40000

        @staticmethod
        def get_hyperparameters():
            return {
                "learning_rate": Config.NN.LEARNING_RATE,
                "gamma": Config.NN.GAMMA,
                "max_memory": Config.NN.MAX_MEMORY,
                "batch_size": Config.NN.BATCH_SIZE,
                "epsilon_start": Config.NN.EPSILON_START,
                "epsilon_end": Config.NN.EPSILON_END,
                "epsilon_decay": Config.NN.EPSILON_DECAY,
                "update_target_every": Config.NN.UPDATE_TARGET_EVERY,
                "tau": Config.NN.TAU,
                "alpha": Config.NN.ALPHA,
                "beta_start": Config.NN.BETA_START,
                "beta_max": Config.NN.BETA_MAX,
                "beta_increment_steps": Config.NN.BETA_INCREMENT_STEPS
            }

        class Arch:
            INPUTS_DESC: list[str] = ["section_rel_x", "section_rel_y", "next_turn" ,"velocity", "acceleration", "relative_yaw", "second_edge_length", "second_turn", "third_edge_length", "third_turn"]
            OUTPUTS_DESC: list[str] = ["release", "forward", "right", "left", "forward_right", "forward_left"]
            ACTIVATED_KEYS_PER_OUTPUT: list[tuple[int]] = [(1, 1, 1, 1), (1, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0), (1, 0, 0, 1), (1, 1, 0, 0)]
            REWARD_DESC: str = "distance travelled projected on the section's x axis (progression on the track)"

            INPUT_SIZE: int = len(INPUTS_DESC)
            OUTPUT_SIZE: int = len(OUTPUTS_DESC)

            LAYER_SIZES: list[int] = [256, 128]
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