import random
import numpy as np

from tminterface.client import Client
from tminterface.interface import TMInterface
from collections import deque
import shutil

from ..map_interaction.map_layout import MapLayout
from ..utils.utils import *
from .model import Model, QTrainer
from ..config import Config
from ..utils.tm_logger import TMLogger

class HorizonClient(Client):
    def __init__(self, num, shared_dict, model_path=None, init_iterations=0) -> None:
        super(HorizonClient, self).__init__()
        self.num = num
        self.map_layout = MapLayout()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = Model().to(self.device)
        self.trainer = QTrainer(self.model, self.device, Config.NN.LEARNING_RATE, Config.NN.GAMMA)

        self.memory = deque(maxlen=Config.NN.MAX_MEMORY)
        self.reward = 0.0
        self.epsilon = Config.NN.EPSILON_START
        self.prev_position = None
        self.prev_positions = deque(maxlen=5 * Config.Game.NUMBER_OF_ACTIONS_PER_SECOND)        # 5-second long memory
        self.current_state = None
        self.prev_game_state: StateAction = None
        self.iterations = init_iterations
        self.ready = False

        self.logger = TMLogger(get_device_info(self.device.type))
        self.rewards_queue = shared_dict["reward"]
        self.epsilon_dict = shared_dict["epsilon"]
        self.manual_epsilon = None

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")

        self.model.train()

    def __str__(self) -> str:
        return f"x position: {self.current_state[0].item():<8.2f} y position: {self.current_state[1].item():<8.2f} next turn: {self.current_state[2].item():<8} yaw: {self.current_state[3].item():<8.2f}"

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")

    def launch_map(self, iface: TMInterface) -> None:
        iface.execute_command(f"map {get_default_map()}")
        iface.set_speed(Config.Game.GAME_SPEED)

    def save_model(self) -> None:
        self.logger.update_log_id()
        directory = self.logger.dump()

        if not directory:
            print("Failed to save the model")
            return

        model_path = os.path.join(directory, "model.pth")
        torch.save(self.model.state_dict(), model_path)
        
        # Copy the model dir contents to the latest model dir
        for file in os.listdir(directory):
            shutil.copy(os.path.join(directory, file), Config.Paths.LATEST_MODEL_PATH)

        print(f"Model saved to {model_path} and in the latest model directory")


    def get_state(self, iface: TMInterface):
        state = iface.get_simulation_state()

        section_rel_pos, next_turn = self.map_layout.get_section_info(state.position[0], state.position[2])
        relative_yaw = self.map_layout.get_car_orientation(state.yaw_pitch_roll[0], state.position[0], state.position[2])

        current_state = torch.tensor([
            section_rel_pos[0],
            section_rel_pos[1],
            next_turn,
            state.display_speed / 999,
            relative_yaw
        ], dtype=torch.float, device=self.device)

        return current_state
    
    def update_epsilon(self):
        if self.epsilon_dict["manual"]:
            self.epsilon = self.epsilon_dict["value"]
        else:
            self.epsilon = Config.NN.EPSILON_END + (Config.NN.EPSILON_START - Config.NN.EPSILON_END) * np.exp(-1. * self.iterations / Config.NN.EPSILON_DECAY) 
            self.epsilon_dict["value"] = self.epsilon

    def get_action(self, state) -> torch.Tensor:
        self.update_epsilon()

        if random.random() < self.epsilon:
            return torch.randint(0, Config.NN.Arch.OUTPUT_SIZE, (), device=self.device)
        else:
            with torch.no_grad():
                return torch.argmax(self.model(state.to(self.device)))


    def send_input(self, iface: TMInterface, move) -> None:
        match move: 
            case 0:
                iface.set_input_state(accelerate=False, left=False, right=False)
            case 1:
                iface.set_input_state(accelerate=True, left=False, right=False)
            case 2:
                iface.set_input_state(accelerate=False, left=False, right=True)
            case 3:
                iface.set_input_state(accelerate=False, left=True, right=False)
            case 4:
                iface.set_input_state(accelerate=True, left=False, right=True)
            case 5:
                iface.set_input_state(accelerate=True, left=True, right=False)
            case _:
                iface.set_input_state(accelerate=False, left=False, right=False)

    def get_reward(self, iface: TMInterface):
        if self.prev_position is None:
            return torch.tensor(0, device=self.device)
        current_position = iface.get_simulation_state().position[0], iface.get_simulation_state().position[2]
        current_reward = self.map_layout.get_distance_reward(self.prev_position, current_position)
        return torch.tensor(current_reward, device=self.device)

    def determine_done(self, iface: TMInterface):
        state = iface.get_simulation_state()

        if state.position[1] < 23: # If the car is below the track
            return torch.tensor(1.0, device=self.device, dtype=torch.float)
        if not state.player_info.finish_not_passed:
            return torch.tensor(1.0, device=self.device, dtype=torch.float)
        if self.prev_positions and len(self.prev_positions) == 50 and np.linalg.norm(np.array(self.prev_positions[0]) - np.array(self.prev_positions[-1])) < 5:    # If less than 5 meters were travelled in the last 5 seconds
            return torch.tensor(1.0, device=self.device, dtype=torch.float)
        #if state.display_speed < 5 and state.race_time > 2000:
        #    return torch.tensor(True, device=self.device)
        return torch.tensor(0.0, device=self.device, dtype=torch.float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > Config.NN.BATCH_SIZE:
            mini_sample = random.sample(self.memory, Config.NN.BATCH_SIZE)        # List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states =  torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.stack(dones).to(self.device)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def on_run_step(self, iface: TMInterface, _time: int) -> None:
        if _time == 0:
            self.ready = True

        if _time >= 0 and _time % Config.Game.INTERVAL_BETWEEN_ACTIONS == 0 and self.ready:
            start_time = time.time()
            self.current_state = self.get_state(iface)  # Get the current state
            done = self.determine_done(iface)

            if self.prev_game_state is not None:         # If this is not the first state, train the model
                current_reward = self.get_reward(iface)
                if not self.epsilon_dict["manual"]:
                    self.remember(self.prev_game_state.state, self.prev_game_state.action, current_reward, self.current_state, done)
                    self.train_short_memory(self.prev_game_state.state, self.prev_game_state.action, current_reward, self.current_state, done)
                self.reward += current_reward.item()

            action = self.get_action(self.current_state)                                    # Get the action
            self.prev_game_state = StateAction(self.current_state, action)       # Save the current state and action for the next iteration
            self.prev_position = iface.get_simulation_state().position[0], iface.get_simulation_state().position[2] # Save the previous position for the reward's calculation
            self.prev_positions.append(self.prev_position)

            self.send_input(iface, action)                # Send the action to the game

            end_time = time.time()
            total_time = end_time - start_time
            if total_time * 1000 > Config.Game.INTERVAL_BETWEEN_ACTIONS / Config.Game.GAME_SPEED:
                print(f"Warning: the action took {total_time * 1000:.2f}ms to execute, it should've taken less than {Config.Game.INTERVAL_BETWEEN_ACTIONS / Config.Game.GAME_SPEED:.2f}ms")

            if done:
                self.ready = False
                if not self.epsilon_dict["manual"]:
                    self.iterations += 1
                    self.rewards_queue.put(self.reward)
                    self.logger.add_run(self.iterations, _time, self.reward)
                    self.train_long_memory()
                print(f"Iteration: {self.iterations:<8} reward: {self.reward:<8.2f} epsilon: {self.epsilon:<8.3f}")    
                self.reward = 0.0
                self.prev_position = None
                self.prev_positions.clear()
                self.prev_game_state = None
                iface.horn()
                iface.respawn()

class StateAction:
    def __init__(self, state, action):
        self.state = state
        self.action = action