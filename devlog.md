# Devlog

### 2-6 March 2025, 7h

Launch multiple clients at the same time, logging in, launching the map and focussing the windows

---

### 10 March 2025, 10h

- Created the first map layout
- Created the agent, using as inputs: the absolute position of the agent, the absolute yaw, the next turn's block position and its direction. The reward is simply the block index the car is on.

---

### 11 March 2025, 5h

- Better parsed the map to extract road sections and turns. A section is a portion of the road between two turns.
- Positioned the agent relatively to the road section it is on. That means the state used the relative position of the car on the section instead of the absolute position. Additionally, we still use the next_turn as input and the absolute yaw.

---

### 12 March 2025, 6h

- Updated the reward. Now it is the distance that was traveled from the previous state to the current state.
- Changed the evolution of epsilon to an exponential decay.

#### Additional notes

- Fixed the bug that was causing to have empty sections.
- Fixed a bug that would cause the agent to get a massive negative reward upon respawning. This was due to the fact that we didn't reset the previous state (so the previous position) before the agent respawned.
- Tried to plot reward in real time but **failed**

---

### 13 March 2025

- First run of the agent on the map. Passed the first two turns and trained for 7h (22000 iterations) see the [model](models/1st_run_13_03.pth).
- Refactored the utils file into two classes: TMLoader and MapLayout.
- Changed the yaw input to be relative to the road section. It is in [-1, 1] where 0 is the car facing the road, -0.5 is the car facing the left side of the road and 0.5 is the car facing the right side of the road.
- Removed fps cap
