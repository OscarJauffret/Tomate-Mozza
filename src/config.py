
class Config:
    class Paths:
        MAP_LAYOUT_PATH: str = "maps/ordered_blocks.json"
        MAP_GBX_OUTPUT_PATH: str = "maps/horizon_layout.txt"

    class Game:
        NUMBER_OF_CLIENTS: int = 1
        WINDOW_NAME: str = "TrackMania Nations Forever (TMInterface 1.4.3)"

    class NN:
        INPUT_SIZE: int = 4
        OUTPUT_SIZE: int = 6
        LEARNING_RATE: float = 0.001
        GAMMA: float = 0.9
        MAX_MEMORY: int = 100_000
        BATCH_SIZE: int = 128
        EPSILON_START: float = 0.9
        EPSILON_END: float = 0.05
        EPSILON_DECAY: int = 2000