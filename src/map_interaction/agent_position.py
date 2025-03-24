import json
from ..config import Config
from typing import Tuple, List

class AgentPosition:

    def __init__(self) -> None:
        """
        Initialize the AgentPosition object by loading the map layout from the json file
        """
        with open(Config.Paths.MAP_BLOCKS_PATH, "r") as f:
            data = json.load(f)
            # Convert lists to tuples for nodes and turns
            self.nodes: List[Tuple[int, int]] = [tuple(node) for node in data["nodes"]]
            self.turns: List[int] = data["turns"]


    def _get_closest_edge(self, agent_block_position: Tuple[float, float]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Get the closest edge to the agent
        :param agent_block_position: the block position of the agent
        :return: the closest edge to the agent as a tuple of two node tuples
        """
        u, v = agent_block_position
        min_dist_sq = float("inf")
        closest_edge = ((-1, -1), (-1, -1))
        
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
                closest_edge = (self.nodes[i], self.nodes[i + 1])

        return closest_edge


    @staticmethod
    def _absolute_position_to_block_position(agent_absolute_position: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert the absolute position of the agent to the block position
        :param agent_absolute_position: the absolute position of the agent
        :return: the block position of the agent
        """
        return (agent_absolute_position[0] / Config.Game.BLOCK_SIZE, 
                agent_absolute_position[1] / Config.Game.BLOCK_SIZE)
    

    def _get_relative_position(self, agent_block_position: Tuple[float, float], 
                              closest_edge: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[float, float]:
        """
        Get the relative position of the agent on the track
        :param agent_block_position: the block position of the agent
        :param closest_edge: the closest edge to the agent
        :return: the relative position of the agent (x, y) where:
                 x: position along the edge (0 = start, 1 = end)
                 y: perpendicular distance from the edge (-1 = left, 0 = center, 1 = right)
        """
        if closest_edge == ((-1, -1), (-1, -1)):
            return (-1, -1)  # No closest edge found

        start_node, end_node = closest_edge
        start_x, start_y = start_node
        end_x, end_y = end_node
        
        # Edge vector
        edge_vector = (end_x - start_x, end_y - start_y)
        edge_length = abs(edge_vector[0]) if edge_vector[1] == 0 else abs(edge_vector[1])
        
        # Normalize edge vector
        if edge_length > 0:
            normalized_edge = (edge_vector[0] / edge_length, edge_vector[1] / edge_length)
        else:
            return (-1, -1)  # Edge has zero length
        
        # Vector from start to agent
        agent_vector = (agent_block_position[0] - start_x, agent_block_position[1] - start_y)
        
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
        
        return (x_relative, y_relative)

    
    def get_distance_reward(self, previous_absolute_pos: Tuple[float, float], 
                           current_absolute_pos: Tuple[float, float]) -> float:
        """
        Calculate the distance reward between two positions
        If the agent is on the same edge, the reward is the distance between the two relative positions
        If the agent is on different edges, the reward is the distance from the previous agent position 
            to the corner node in the relative coordinates of the previous edge + the distance from 
            the corner node to the current position of the agent in the relative coordinates of the current edge.
        :param previous_absolute_pos: the previous absolute position of the agent
        :param current_absolute_pos: the current absolute position of the agent
        :return: the distance reward
        """
        prev_agent_block_position = self._absolute_position_to_block_position(previous_absolute_pos)
        cur_agent_block_position = self._absolute_position_to_block_position(current_absolute_pos)
        
        prev_closest_edge = self._get_closest_edge(prev_agent_block_position)
        cur_closest_edge = self._get_closest_edge(cur_agent_block_position)
        
        prev_relative_pos = self._get_relative_position(prev_agent_block_position, prev_closest_edge)
        cur_relative_pos = self._get_relative_position(cur_agent_block_position, cur_closest_edge)

        if prev_relative_pos == (-1, -1) or cur_relative_pos == (-1, -1):
            return 0

        prev_x, _ = prev_relative_pos
        cur_x, _ = cur_relative_pos

        # Calculate the distance reward
        if prev_closest_edge == cur_closest_edge:  # Same edge
            return cur_x - prev_x
        else:
            # Different edge - handle corner cases
            if prev_closest_edge[1] == cur_closest_edge[0]:  # Normal corner transition
                edge_1_second_node_rel_pos = self._get_relative_position(prev_closest_edge[1], prev_closest_edge)
                edge_2_first_node_rel_pos = self._get_relative_position(cur_closest_edge[0], cur_closest_edge)
                return edge_1_second_node_rel_pos[0] - prev_x + cur_x - edge_2_first_node_rel_pos[0]
            elif prev_closest_edge[0] == cur_closest_edge[1]:  # Backward corner transition
                edge_1_first_node_rel_pos = self._get_relative_position(prev_closest_edge[0], prev_closest_edge)
                edge_2_second_node_rel_pos = self._get_relative_position(cur_closest_edge[1], cur_closest_edge)
                return edge_1_first_node_rel_pos[0] - prev_x + cur_x - edge_2_second_node_rel_pos[0]
            else:
                # Different edges with no connection
                return 0


if __name__ == "__main__":
    agent_position = AgentPosition()
    prev = (17.9 * 32, 14.2 * 32)  # Tuple instead of list
    cur = (17.9 * 32, 14 * 32)     # Tuple instead of list
    print(agent_position.get_distance_reward(prev, cur))