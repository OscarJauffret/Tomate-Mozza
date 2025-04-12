import json
import os
from datetime import datetime
import numpy as np

from ..config import Config

class TMLogger:
    def __init__(self, device: str , algorithm: str):
        """
        Constructor for TMLogger class
        :param device: The device used for training
        :param algorithm: The algorithm used for training
        """
        self.log_id: str = datetime.now().strftime(Config.DATETIME_FORMAT)
        self.map_info: dict = Config.Paths.get_map()
        self.algorithm: str = algorithm
        self.architecture: dict = Config.Arch().get_architecture_description()
        self.training_device: str = device
        self.run_stats: list[_RunStats] = []

        if self.algorithm == "DQN":
            self.hyperparameters: dict = Config.DQN().get_hyperparameters()
        elif self.algorithm == "PPO":
            self.hyperparameters: dict = Config.PPO().get_hyperparameters()

    def update_log_id(self):
        """
        Update the log id to the current time
        :return: None
        """
        self.log_id = datetime.now().strftime(Config.DATETIME_FORMAT)

    def add_run(self, iteration, run_time, reward):
        """
        Add a run to the logger
        :param iteration: The iteration of the run
        :param run_time: The time taken for the run
        :param reward: The reward obtained from the run
        """
        self.run_stats.append(_RunStats(iteration, run_time, reward))

    def _compute_stats(self):
        """
        Compute the statistics of the runs
        :return: A dictionary containing the statistics
        """
        total_num_runs = len(self.run_stats)
        training_time_hours = self._get_training_time('h')
        training_time_minutes = self._get_training_time('ms')
        average_reward = self._compute_average_reward()
        best_reward = self._compute_best_reward()
        recent_average_reward_percentage = 0.1
        recent_average_reward = self._compute_recent_average_reward(recent_average_reward_percentage)
        lowest_rewards_percentage = 0.1
        lowest_rewards_average = self._average_low_rewards(recent_average_reward_percentage, lowest_rewards_percentage)
        recent_quantiles_reward = self._compute_recent_quantiles_reward(recent_average_reward_percentage)
        return {
            "total number of runs": total_num_runs,
            "training time": {"hours": training_time_hours, "milliseconds": training_time_minutes},
            "average reward": average_reward,
            "best reward": best_reward,
            "recent average reward": {"percentage of runs considered": recent_average_reward_percentage,
                                      "number of runs considered": int(np.ceil(total_num_runs * recent_average_reward_percentage)),
                                      "average reward": recent_average_reward},
            "lowest rewards average": {"percentage of runs considered": recent_average_reward_percentage,
                                       "percentage of low runs considered among the recent ones": lowest_rewards_percentage,
                                       "number of runs considered": int(np.ceil(total_num_runs * recent_average_reward_percentage * lowest_rewards_percentage)),
                                       "average reward": lowest_rewards_average},
            "recent quartiles reward": {"percentage of runs considered": recent_average_reward_percentage,
                                        "number of runs considered": int(np.ceil(total_num_runs * recent_average_reward_percentage)),
                                        "first quartile": recent_quantiles_reward[0],
                                        "second quartile": recent_quantiles_reward[1],
                                        "third quartile": recent_quantiles_reward[2]}


        }

    def _get_training_time(self, scale):
        """
        Get the total training time
        :param scale: The scale of the time. 'h' for hours, 'm' for minutes, 's' for seconds, 'ms' for milliseconds
        :return: The total training time
        """
        total_time = sum([run.run_time for run in self.run_stats])
        if scale == 'h':
            return total_time / (3600 * 1000)
        if scale == 'm':
            return total_time / (60 * 1000)
        if scale == 's':
            return total_time / 1000
        if scale == 'ms':
            return total_time
        return total_time

    def _compute_average_reward(self):
        """
        Compute the average reward of all the runs
        :return: The average reward
        """
        if len(self.run_stats) == 0:
            return 0
        return sum([run.reward for run in self.run_stats]) / len(self.run_stats)

    def _compute_best_reward(self):
        """
        Compute the best reward of all the runs
        :return: The best reward
        """
        if len(self.run_stats) == 0:
            return 0
        return max([run.reward for run in self.run_stats])

    def _compute_recent_average_reward(self, percentage):
        """
        Compute the average reward of the most recent runs
        :param percentage: The percentage of runs to consider
        :return: The average reward of the most recent runs
        """
        num_runs = int(np.ceil(len(self.run_stats) * percentage))
        return sum([run.reward for run in self.run_stats[-num_runs:]]) / num_runs

    def _average_low_rewards(self, percentage, low_percentage):
        """
        Compute the average of the lowest rewards among the recent runs
        :param percentage: The percentage of runs to consider
        :param low_percentage: The percentage of low runs to consider among the recent runs
        :return: The average of the lowest rewards
        """
        num_runs = int(np.ceil(len(self.run_stats) * percentage))
        num_low_runs = int(np.ceil(num_runs * low_percentage))
        low_runs = sorted([run.reward for run in self.run_stats[-num_runs:]])[:num_low_runs]
        return sum(low_runs) / num_low_runs

    def _compute_recent_quantiles_reward(self, percentage):
        """
        Compute the three quantiles of the most recent runs
        :param percentage: The percentage of runs to consider
        :return: The quantiles of the most recent runs
        """
        num_runs = int(np.ceil(len(self.run_stats) * percentage))
        recent_runs = sorted([run.reward for run in self.run_stats[-num_runs:]])
        return np.quantile(recent_runs, [0.25, 0.5, 0.75])

    def dump(self):
        """
        Dump the log to the log file
        :return: The directory where the log file is saved
        """
        if not self.run_stats:
            return None

        directory = os.path.join(Config.Paths().MODELS_PATH, self.log_id)

        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, Config.Paths.STAT_FILE_NAME)

        with open(file_path, "w") as f:
            log ={
                "algorithm": self.algorithm,
                "map_info": self.map_info,
                "hyperparameters": self.hyperparameters,
                "device": self.training_device,
                "architecture": self.architecture,
                "statistics": self._compute_stats(),
                "runs": [run.get_stats() for run in self.run_stats]
            }
            json.dump(log, f, indent=4)

        return directory

    def load(self, log_path: str) -> None:
        """
        Load the log from a log file
        :param log_path: The path to the log file
        :return: None
        """
        with open(log_path, "r") as f:
            log = json.load(f)
            self.map_info = log["map_info"]
            self.hyperparameters = log["hyperparameters"]
            self.training_device = log["device"]
            self.architecture = log["architecture"]
            self.run_stats = [_RunStats(run["iteration"], run["run_time"], run["reward"]) for run in log["runs"]]


class _RunStats:
    def __init__(self, iteration, run_time, reward):
        """
        Constructor for RunStats class
        :param iteration: The iteration of the run
        :param run_time: The time taken for the run
        :param reward: The reward obtained from the run
        """
        self.iteration = iteration
        self.run_time = run_time
        self.reward = reward

    def get_stats(self):
        """
        Get the statistics of the run
        :return: A dictionary containing the run statistics
        """
        return {
            "iteration": self.iteration,
            "run_time": self.run_time,
            "reward": self.reward
        }