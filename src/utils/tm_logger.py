import json
import os
from datetime import datetime

from ..config import Config

class TMLogger:
    def __init__(self, device):
        """
        Constructor for TMLogger class
        :param device: The device used for training
        """
        self.log_id: str = datetime.now().strftime(Config.DATETIME_FORMAT)
        self.hyperparameters: dict = Config.NN().get_hyperparameters()
        self.training_device: str = device
        self.run_stats: list[RunStats] = []

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

    def dump(self):
        """
        Dump the log to the log file
        :return: The directory where the log file is saved
        """
        directory = os.path.join(Config.Paths().MODELS_PATH, self.log_id)

        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, f"{self.log_id}.json")

        with open(file_path, "w") as f:
            log ={
                "hyperparameters": self.hyperparameters,
                "device": self.training_device,
                "runs": [run.get_stats() for run in self.run_stats]
            }
            json.dump(log, f, indent=4)

        return directory


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