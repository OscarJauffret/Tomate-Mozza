import ast
import config
blocks = []

names_lut = {"StadiumRoadMainStartLine": "Start", "StadiumPlatformToRoadMain": "Transition", "StadiumRoadMainFinishLine": "Finish", "StadiumCircuitBase": "Road"}

def is_next_to(pos1, pos2):
    return abs(pos1[0] - pos2[0]) == 1 and abs(pos1[1] - pos2[1]) == 0 or abs(pos1[0] - pos2[0]) == 0 and abs(pos1[1] - pos2[1]) == 1

with open(config.MAP_GBX_OUTPUT_PATH, "r") as f:
    for line in f:
        if line.startswith("Flags") or line.startswith("Rotation"):
            continue
        if line.startswith("Name"):
            name = line.split(" ")[1].strip()
            blocks.append([names_lut.get(name, name)])
        elif line.startswith("Position"):
            position = line.split(" ", maxsplit=1)[-1].strip()
            blocks[-1].append(ast.literal_eval(position))

blocks = sorted(blocks, key=lambda x: x[1][1])  # Sort by height

# Only keep the blocks that are the highest in their column
highest_blocks = []

for i, block in enumerate(blocks):
    without_y = [block[1][0], block[1][2]]
    if i == 0:
        highest_blocks.append([block[0], without_y])

    elif without_y not in [b[1] for b in highest_blocks]:
        highest_blocks.append([block[0], without_y])

start_block = None
transition_block= None
for block in highest_blocks:
    if block[0] == "Start":
        start_block = block[1]
        highest_blocks.remove(block)
        break

for block in highest_blocks:
    if block[0] == "Transition" and is_next_to(start_block, block[1]):
        transition_block = block[1]
        highest_blocks.remove(block)
        break

if not start_block or not transition_block:
    raise Exception("Start or Transition block not found")

ordered_blocks = [start_block, transition_block]

while highest_blocks:
    for i, block in enumerate(highest_blocks):
        if is_next_to(ordered_blocks[-1], block[1]):
            ordered_blocks.append(block[1])
            highest_blocks.remove(block)
            break

with open(config.MAP_LAYOUT_PATH, "w") as f:
    f.write(str(ordered_blocks))