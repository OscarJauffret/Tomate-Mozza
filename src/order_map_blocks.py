import ast
import json

from config import Config


def is_next_to(pos1, pos2):
    """
    Check if two blocks are next to each other (manhattan distance of 1)
    :param pos1: the first block
    :param pos2: the second block
    :return: True if the blocks are next to each other, False otherwise
    """
    return abs(pos1[0] - pos2[0]) == 1 and abs(pos1[1] - pos2[1]) == 0 or abs(pos1[0] - pos2[0]) == 0 and abs(pos1[1] - pos2[1]) == 1

def _read_map_layout(map_name):
    """
    Read the map layout from the output of the GBX file. First you need to read the GBX file and read it using pygbx.
    :return: A list of the blocks of the map, containing, for each block, the name and the position
    """
    blocks = []
    names_lut = {"StadiumRoadMainStartLine": "Start", "StadiumPlatformToRoadMain": "Transition",
                 "StadiumRoadMainFinishLine": "Finish", "StadiumCircuitBase": "Road"}
    with open(map_name, "r") as f:
        for line in f:
            if line.startswith("Flags") or line.startswith("Rotation"):
                continue
            if line.startswith("Name"):
                name = line.split(" ")[1].strip()
                blocks.append([names_lut.get(name, name)])
            elif line.startswith("Position"):
                position = line.split(" ", maxsplit=1)[-1].strip()
                blocks[-1].append(ast.literal_eval(position))
    return blocks


def _keep_highest_blocks(blocks):
    """
    Keep only the blocks that are the highest in their column
    :param blocks: the list of blocks
    :return: the list of blocks that are the highest in their column
    """
    blocks = sorted(blocks, key=lambda x: x[1][1])  # Sort by height
    highest_blocks = []

    for i, block in enumerate(blocks):
        without_y = [block[1][0], block[1][2]]
        if i == 0:
            highest_blocks.append([block[0], without_y])

        elif without_y not in [b[1] for b in highest_blocks]:
            highest_blocks.append([block[0], without_y])

    return highest_blocks

def _order_blocks_starting_with_start_block(blocks):
    """
    Order the blocks starting with the start block and then the transition block (to be sure to go in the right direction at the beginning)
    :param blocks: the list of blocks containing their name and position
    :return: the ordered list of blocks
    """
    start_block = None
    transition_block= None
    for block in blocks:
        if block[0] == "Start":
            start_block = block[1]
            blocks.remove(block)
            break

    for block in blocks:
        if block[0] == "Transition" and is_next_to(start_block, block[1]):
            transition_block = block[1]
            blocks.remove(block)
            break

    if not start_block or not transition_block:
        raise Exception("Start or Transition block not found")

    ordered_blocks = [start_block, transition_block]

    while blocks:
        for i, block in enumerate(blocks):
            if is_next_to(ordered_blocks[-1], block[1]):
                ordered_blocks.append(block[1])
                blocks.remove(block)
                break

    return ordered_blocks

def order_blocks_of_map(map_name):
    """
    Order the blocks of a map. First you need to read the GBX file and read it using pygbx.
    :param map_name: the name of the file containing the map
    :return: the ordered list of blocks
    """
    blocks = _read_map_layout(map_name)
    blocks = _keep_highest_blocks(blocks)
    blocks = _order_blocks_starting_with_start_block(blocks)

    return blocks


def calculate_direction(block1, block2):
    """
    Calculate the direction between two blocks
    :param block1: the first block
    :param block2: the second block
    :return: the direction between the two blocks
    """
    return (block2[0] - block1[0], block2[1] - block1[1])


def get_sections_and_turns(blocks):
    """
    Get the sections and turns of a map. A section is a group of blocks that form a straight line.
    :param blocks: the list of blocks (ordered)
    :return: a tuple containing (the list of sections, the list of turns)
    """
    sections = []
    turns = []
    current_section = [blocks[0], blocks[1]]
    current_direction = calculate_direction(current_section[0], current_section[1])
    for i in range(2, len(blocks)):
        new_direction = calculate_direction(current_section[-1], blocks[i])
        if new_direction == current_direction:
            current_section.append(blocks[i])
        else:
            vector_product = current_direction[0] * new_direction[1] - current_direction[1] * new_direction[0]
            if vector_product > 0:
                turn = 1
            else:
                turn = -1
            turns.append(turn)
            current_section = [current_section[0], current_section[-1]]
            sections.append(current_section)
            current_section = [blocks[i]]
            current_direction = new_direction

    current_section = [current_section[0], current_section[-1]]
    sections.append(current_section)
    return sections, turns


def write_map_layout(blocks, sections, turns):
    """
    Write the map layout to a file
    :param blocks: the list of blocks
    :param sections: the list of sections
    :param turns: the list of turns
    """
    map_layout = {
        "layout": blocks,
        "sections": sections,
        "turns": turns
    }

    with open(Config.Paths.MAP_LAYOUT_PATH, "w") as f:
        f.write(json.dumps(map_layout))


blocks = order_blocks_of_map(Config.Paths.MAP_GBX_OUTPUT_PATH)
sections, turns = get_sections_and_turns(blocks)
write_map_layout(blocks, sections, turns)