import json
import numpy as np
from ..config import Config

class MapLayout:

    def __init__(self) -> None:
        """
        Initialize the MapLayout object by loading the map layout from the json file
        """
        self.rotation_angles_from_absolute_to_section_coordinates = {(1, 0): 0, (0, 1): np.pi / 2, (-1, 0): np.pi, (0, -1): 3 * np.pi / 2}
        with open(Config.Paths.MAP_LAYOUT_PATH, "r") as f:
            data = json.load(f)
            self.blocks: np.ndarray[np.ndarray[int]] = np.array(data["layout"])
            self.sections: np.ndarray[np.ndarray[tuple[int, int]]] = np.array(data["sections"])
            self.turns: np.ndarray[float] = data["turns"]

    def _is_in_section(self, block_x: int, block_y: int, section: np.ndarray[tuple[int, int]]) -> bool:
        """
        Check if a block is in the given section
        :param block_x: The x coordinate of the block
        :param block_y: The y coordinate of the block
        :param section: The section to check
        :return: True if the block is in the section, False otherwise
        """
        [start_x, start_y], [end_x, end_y] = section
        start_x, end_x = min(start_x, end_x), max(start_x, end_x)
        start_y, end_y = min(start_y, end_y), max(start_y, end_y)
        return start_x <= block_x <= end_x and start_y <= block_y <= end_y

    def get_current_block(self, pos_x: int, pos_y: int) -> tuple[int, int]:
        """
        Get the block in which the car is located
        :param pos_x: The x position of the car
        :param pos_y: The y position of the car
        :return: The block in which the car is located as a tuple (block_x, block_y)
        """
        block_x = np.floor(pos_x / 32)
        block_y = np.floor(pos_y / 32)
        return block_x, block_y

    def _get_section_dimension(self, section_index: int) -> np.array:
        """
        Get the dimensions of the section
        :param section_index: The index of the section
        :return: The dimensions of the section as a numpy array [length, width]
        """
        start_x, start_y = self.sections[section_index][0]
        end_x, end_y = self.sections[section_index][1]
        return np.array([(abs(start_x - end_x) + abs(start_y - end_y)) * 32, 16])

    def get_distance_reward(self, prev_pos: tuple[int, int], current_pos: tuple[int, int]) -> float:
        """
        Get the reward based on the distance between the previous and current position. The reward is greater if the
        distance is larger, and the reward is negative if the car is moving backwards.
        :param prev_pos: The previous position of the car as a tuple (x, y) (absolute)
        :param current_pos: The current position of the car as a tuple (x, y) (absolute)
        :return: The reward based on the distance between the previous and current position
        """
        prev_pos_in_section, prev_section_index = self._get_position_relative_to_section(*prev_pos)
        current_pos_in_section, current_section_index = self._get_position_relative_to_section(*current_pos)

        prev_section_dimension = self._get_section_dimension(prev_section_index)
        current_section_dimension = self._get_section_dimension(current_section_index)

        if prev_section_index == -1 or current_section_index == -1:
            return 0

        if prev_section_index == current_section_index:
            mul = -1 if current_pos_in_section[0] < prev_pos_in_section[0] else 1
            # Just consider the x distance (we want to reward the car for progressing on the track, we don't care that it's doing zigzags)
            dist = np.abs(current_pos_in_section[0] - prev_pos_in_section[0]) * current_section_dimension[0]
            return mul * dist

        mul = -1 if current_section_index < prev_section_index else 1
        if prev_pos_in_section[0] <= 1:
            remaining_previous_section_distance = np.abs(1 - prev_pos_in_section[0]) * prev_section_dimension[0]
        else:
            remaining_previous_section_distance = 0
        y_section_distance = Config.Game.BLOCK_SIZE // 2
        current_section_distance = np.abs(current_pos_in_section[0]) * current_section_dimension[0]
        dist = remaining_previous_section_distance + y_section_distance + current_section_distance

        return mul * dist

    def _get_position_relative_to_section(self, pos_x: int, pos_y: int) -> tuple[tuple[int, int], int]:
        """
        Get the relative position of the car in the section and the index of the section
        :param pos_x: The x position of the car (absolute)
        :param pos_y: The y position of the car (absolute)
        :return: A tuple containing the relative position of the car in the section and the index of the section. [-1, -1], -1 if the car is not in a section
        """
        section, section_index = self._get_current_section(pos_x, pos_y)
        if np.all(section == np.array([(-1, -1), (-1, -1)])):
            return (-1, -1), -1
        [start_x, start_y], [end_x, end_y] = section
        # Scale the bounds to the absolute position, and add half a block to move the origin to the center of the block
        start_x = start_x * Config.Game.BLOCK_SIZE + Config.Game.BLOCK_SIZE // 2
        start_y = start_y * Config.Game.BLOCK_SIZE + Config.Game.BLOCK_SIZE // 2
        end_x = end_x * Config.Game.BLOCK_SIZE + Config.Game.BLOCK_SIZE // 2
        end_y = end_y * Config.Game.BLOCK_SIZE + Config.Game.BLOCK_SIZE // 2

        # To change the coordinate system to the "section" coordinate system, we need to translate and rotate the absolute coordinate system
        # Translate the position to the origin of the section
        translated_x = pos_x - start_x
        translated_y = pos_y - start_y

        # Get the direction of the section
        direction = self._get_direction_of_section(section, False)
        normalized_direction = self._get_direction_of_section(section, True)

        # Get the angle by which we need to rotate the coordinate system
        angle = self.rotation_angles_from_absolute_to_section_coordinates[normalized_direction]

        # Rotate the coordinate system
        rotated_x = translated_x * np.cos(-angle) - translated_y * np.sin(-angle)
        rotated_y = translated_x * np.sin(-angle) + translated_y * np.cos(-angle)

        # Get the normalized position
        size = np.linalg.norm(np.array(direction)), Config.Game.BLOCK_SIZE // 2
        normalized_x = rotated_x / size[0]
        normalized_y = rotated_y / size[1]
        
        return (normalized_x, normalized_y), section_index

    def get_section_info(self, pos_x, pos_y):
        """
        Get the section information that is passed in the state of the client. The info contains the relative position of the
        car in the section and the next turn.
        :param pos_x: The x position of the car (absolute)
        :param pos_y: The y position of the car (absolute)
        :return: A tuple containing the relative position of the car in the section and the next turn
        """
        pos_in_section, section_index = self._get_position_relative_to_section(pos_x, pos_y)
        if pos_in_section == (-1, -1):
            return (-1, -1), 0
        if self.turns:
            next_turn = self.turns[section_index]
        else:
            next_turn = 0
        return pos_in_section, next_turn

    @staticmethod
    def _get_direction_of_section(section: np.ndarray[tuple[int, int]], normalized) -> tuple[int, int]:
        """
        Get the direction of the current section
        :param section: The section to get the direction of
        :param normalized: Whether to normalize the direction. If not normalized, the direction is returned in absolute coordinates (multiples of 32)
        :return: The direction of the current section as a tuple (x, y), normalized if specified
        """
        [start_x, start_y], [end_x, end_y] = section
        direction = ((end_x - start_x) * Config.Game.BLOCK_SIZE, (end_y - start_y) * Config.Game.BLOCK_SIZE)
        if normalized:
            direction_norm = [0, 0]
            for i in range(2):
                direction_norm[i] = 0 if direction[i] == 0 else int(direction[i] / abs(direction[i]))
            return tuple(direction_norm)
        return direction

    def _get_current_section(self, pos_x: int, pos_y: int) -> tuple[np.ndarray[tuple[int, int]], int]:
        """
        Get the bounds of the current section
        :param pos_x: The x position of the car (absolute)
        :param pos_y: The y position of the car (absolute)
        :return: The bounds of the current section as a numpy array [start, end] (in block coordinates) and the index of the section
        """
        block_x, block_y = self.get_current_block(pos_x, pos_y)
        for i, section in enumerate(self.sections):
            if self._is_in_section(block_x, block_y, section):
                return section, i
        return np.array([(-1, -1), (-1, -1)]), -1

    def get_car_orientation(self, yaw: float, pos_x: int, pos_y: int) -> float:
        """
        Get the orientation of the car relative to the section it is in.
        - If the car is facing the same direction as the section, the orientation is 0.
        - If the car is facing the opposite direction as the section, the orientation is pi (or -pi).
        - If the car is facing the right side of the section, the orientation is pi / 2.
        - If the car is facing the left side of the section, the orientation is -pi / 2.
        The angle is then normalized by pi to get values between -1 and 1.
        :param yaw: The yaw of the car (in radians, absolute)
        :param pos_x: The x position of the car (absolute)
        :param pos_y: The y position of the car (absolute)
        :return: The orientation of the car relative to the section it is in.
        """
        direction_to_angle = {(0, -1): np.pi, (-1, 0): -np.pi / 2, (0, 1): 0, (1, 0): np.pi / 2}
        section, _ = self._get_current_section(pos_x, pos_y)
        if np.all(section == np.array([(-1, -1), (-1, -1)])):
            return 0
        direction = self._get_direction_of_section(section, True)
        section_angle = direction_to_angle[direction]
        theta = (section_angle - yaw)
        return ((theta + np.pi) % (2 * np.pi) - np.pi) / np.pi



