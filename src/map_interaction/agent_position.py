from math import sqrt
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
            self.nodes: list[list[int]] = data["nodes"]
            self.turns: list[list[int]] = data["turns"]


    def _get_closest_edge(self, agent_block_position: list[float]) -> list[list[int]]:
        """
        Get the closest edge to the agent
        :param agent_block_position: the block position of the agent
        :return: the closest edge to the agent
        """
        u, v = agent_block_position
        min_dist_sq = float("inf")
        closest_edge = None
        
        for i in range(len(self.nodes) - 1):
            x1, y1 = self.nodes[i]
            x2, y2 = self.nodes[i + 1]

            # Edge vector
            dx, dy = x2 - x1, y2 - y1
            edge_length_sq = dx ** 2 + dy ** 2

            # Vector from the agent to the start of the edge
            px, py = u - x1, v - y1

            # Projection of the agent_to_start vector on the edge vector (clamped to [0, 1])
            if edge_length_sq != 0:
                t = (px * dx + py * dy) / edge_length_sq
                t = max(0, min(1, t))
            else:
                t = 0 # Edge length is 0 (should not happen)

            # Closest point on the edge
            closest_x, closest_y = x1 + t * dx, y1 + t * dy

            # Distance between the agent and the closest point
            dist_sq = (u - closest_x) ** 2 + (v - closest_y) ** 2

            # Update the closest edge
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_edge = [self.nodes[i], self.nodes[i + 1]]

        return closest_edge


    @staticmethod
    def _absolute_position_to_block_position(agent_absolute_position: list[float]) -> list[float]:
        """
        Convert the absolute position of the agent to the block position
        :param agent_absolute_position: the absolute position of the agent
        :return: the block position of the agent
        """
        return [agent_absolute_position[0] / Config.Game.BLOCK_SIZE, agent_absolute_position[1] / Config.Game.BLOCK_SIZE]
    

    @staticmethod
    def calculate_direction(block1: list[int], block2: list[int]) -> list[int]:
        """
        Calculate the direction between two blocks, it will be normalized
        :param block1: the first block
        :param block2: the second block
        :return: the direction between the two blocks
        """
        direction = (block2[0] - block1[0], block2[1] - block1[1])
        if direction[0] != 0:
            direction = (direction[0] // abs(direction[0]), 0)
        elif direction[1] != 0:
            direction = (0, direction[1] // abs(direction[1]))
        return direction


    def get_relative_position(self, agent_absolute_position: list[float]) -> list[float]:
        """
        Get the relative position of the agent on the track
        :param agent_absolute_position: the absolute position of the agent
        :return: the relative position of the agent [x, y] where:
                 x: position along the edge (0 = start, 1 = end)
                 y: perpendicular distance from the edge (-1 = left, 0 = center, 1 = right)
        """
        agent_block_position = self._absolute_position_to_block_position(agent_absolute_position)
        closest_edge = self._get_closest_edge(agent_block_position)

        if closest_edge is None:
            return None

        start_node, end_node = closest_edge
        start_x, start_y = start_node
        end_x, end_y = end_node

        
        # Edge vector
        edge_vector = [end_x - start_x, end_y - start_y]
        edge_length = abs(edge_vector[0]) if edge_vector[1] == 0 else abs(edge_vector[1]) # l
        
        # Normalize edge vector
        if edge_length > 0:
            normalized_edge = [edge_vector[0] / edge_length, edge_vector[1] / edge_length]
        else:
            return [0, 0]  # Edge has zero length
        
        # Vector from start to agent
        agent_vector = [agent_block_position[0] - start_x, agent_block_position[1] - start_y]
        
        # Project agent vector onto edge vector to get longitudinal position
        dot_product = agent_vector[0] * normalized_edge[0] + agent_vector[1] * normalized_edge[1]
        
        # Calculate relative position along edge (x)
        # Map from [start_node - 0.5, end_node + 0.5] to [0, 1]
        x_relative = (dot_product + 0.5) / (edge_length + 1)
        x_relative = max(0, min(1, x_relative))  # Clamp to [0, 1]
        
        # Calculate perpendicular distance (y)
        # Cross product to determine side (left/right)
        cross_product = agent_vector[1] * normalized_edge[0] - agent_vector[0] * normalized_edge[1]
        
        # Map perpendicular distance to [-1, 1]
        # Where -1 means 0.5 units to the left, 1 means 0.5 units to the right
        y_relative = cross_product / 0.5
        y_relative = max(-1, min(1, y_relative))  # Clamp to [-1, 1]
        
        return [x_relative, y_relative]


if __name__ == "__main__":
    agent_position = AgentPosition()
    pos = [18, 16]
    print(agent_position.get_relative_position([pos[0] * Config.Game.BLOCK_SIZE, pos[1] * Config.Game.BLOCK_SIZE]))
    