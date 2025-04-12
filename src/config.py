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
        DQN_MODEL_FILE_NAME: str = "model.pth"
        ACTOR_FILE_NAME: str = "actor.pth"
        CRITIC_FILE_NAME: str = "critic.pth"

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

    class PPO:
        LEARNING_RATE: float = 0.0003
        GAMMA: float = 0.99
        LAMBDA: float = 0.95
        EPSILON: float = 0.2
        C1: float = 1
        C2: float = 0.01
        MEMORY_SIZE: int = 128
        BATCH_SIZE: int = 32
        EPOCHS: int = 4     # Number of times to train on a given memory batch

        @staticmethod
        def get_hyperparameters():
            return {
                "learning_rate": Config.PPO.LEARNING_RATE,
                "gamma": Config.PPO.GAMMA,
                "lambda": Config.PPO.LAMBDA,
                "epsilon": Config.PPO.EPSILON,
                "c1": Config.PPO.C1,
                "c2": Config.PPO.C2,
                "memory_size": Config.PPO.MEMORY_SIZE,
                "batch_size": Config.PPO.BATCH_SIZE,
                "epochs": Config.PPO.EPOCHS,
            }

    class DQN:
        LEARNING_RATE: float = 0.001
        GAMMA: float = 0.97

        MAX_MEMORY: int = 100_000
        MIN_MEMORY: int = 10_000
        BATCH_SIZE: int = 128

        EPSILON_START: float = 0.9
        EPSILON_END: float = 0.05
        EPSILON_DECAY: int = 10000

        UPDATE_TARGET_EVERY: int = 1
        TAU: float = 0.02

        ALPHA: float = 0.7
        BETA_START: float = 0.4
        BETA_MAX: float = 1.0
        BETA_INCREMENT_STEPS: int = 40000

        N_STEPS: int = 25  # 2.5 Seconds

        @staticmethod
        def get_hyperparameters():
            return {
                "learning_rate": Config.DQN.LEARNING_RATE,
                "gamma": Config.DQN.GAMMA,
                "max_memory": Config.DQN.MAX_MEMORY,
                "min_memory": Config.DQN.MIN_MEMORY,
                "batch_size": Config.DQN.BATCH_SIZE,
                "epsilon_start": Config.DQN.EPSILON_START,
                "epsilon_end": Config.DQN.EPSILON_END,
                "epsilon_decay": Config.DQN.EPSILON_DECAY,
                "update_target_every": Config.DQN.UPDATE_TARGET_EVERY,
                "tau": Config.DQN.TAU,
                "alpha": Config.DQN.ALPHA,
                "beta_start": Config.DQN.BETA_START,
                "beta_max": Config.DQN.BETA_MAX,
                "beta_increment_steps": Config.DQN.BETA_INCREMENT_STEPS,
                "n_steps": Config.DQN.N_STEPS
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
                "inputs": Config.Arch.INPUTS_DESC,
                "outputs": Config.Arch.OUTPUTS_DESC,
                "input_size": Config.Arch.INPUT_SIZE,
                "output_size": Config.Arch.OUTPUT_SIZE,
                "layer_sizes": Config.Arch.LAYER_SIZES,
                "number_of_hidden_layers": Config.Arch.NUMBER_OF_HIDDEN_LAYERS,
                "reward_description": Config.Arch.REWARD_DESC
            }