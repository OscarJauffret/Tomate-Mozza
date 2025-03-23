import json
import numpy as np

from ..config import Config

class AgentPosition:

    def __init__(self) -> None:
        """
        Initialize the AgentPosition object by loading the map layout from the json file
        """
        with open(Config.Paths.MAP_BLOCKS_PATH, "r") as f:
            data = json.load(f)
            self.nodes: np.ndarray = np.array(data["nodes"])
            self.turns: np.ndarray = np.array(data["turns"])

        self.edges: np.ndarray = np.array([[self.nodes[i], self.nodes[i+1]] for i in range(len(self.nodes) - 1)])
        print(self.edges)

    def _get_closest_edge(self, agent_block_position: np.ndarray) -> np.ndarray:
        """
        Get the closest edge to the agent
        :param agent_block_position: the absolute position of the agent
        :return: the closest edge to the agent
        """
        distances = [self._distance_to_edge(agent_block_position, edge) for edge in self.edges]
        min_index = np.argmin(distances)
        return self.edges[min_index]

    def _distance_to_edge(self, agent_block_position: np.ndarray, edge: np.ndarray) -> float:
        """
        Get the distance between the agent and an edge
        :param agent_block_position: the absolute position of the agent
        :param edge: the edge
        :return: the distance between the agent and the edge
        """
        pass    # TODO

    @staticmethod
    def _absolute_position_to_block_position(agent_absolute_position: np.ndarray) -> np.ndarray:
        """
        Convert the absolute position of the agent to the block position
        :param agent_absolute_position: the absolute position of the agent
        :return: the block position of the agent
        """
        return agent_absolute_position / Config.Game.BLOCK_SIZE

    def get_relative_position(self, agent_absolute_position: np.ndarray) -> np.ndarray:
        """
        Get the relative position of the agent on the track
        :param agent_absolute_position: the absolute position of the agent
        :return: the relative position of the agent
        """
        # TODO
        agent_block_position = self._absolute_position_to_block_position(agent_absolute_position)
        closest_edge = self._get_closest_edge(agent_block_position)
        return closest_edge
