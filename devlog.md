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

### 13 March 2025, 3h

- First run of the agent on the map. Passed the first two turns and trained for 7h (22000 iterations) see the [model](models/1st_run_13_03.pth).
- Refactored the utils file into two classes: TMLoader and MapLayout.
- Changed the yaw input to be relative to the road section. It is in [-1, 1] where 0 is the car facing the road, -0.5 is the car facing the left side of the road and 0.5 is the car facing the right side of the road.
- Removed fps cap

---

### 14 March 2025, 3h

- Second run with updated yaw input, smaller NN and faster epsilon decay. The agent is not better than the first run after a similar amount of iterations. See the [model](models/2nd_run_14_03.pth).
- Reorganized the repository in packages. This means that to run the code, we now have to use the command `python -m src.main`.
- Added the [TMLogger](src/utils/tm_logger.py) class to log the training process. It logs the hyperparameters, the reward and run time of all the runs, and the device used. It could be interesting to log the model's architecture like the number of layers, neurons and precise inputs (just give the names).
- Added multiprocessing.Queue() to communicate between the client and the main process. Added live plotting of the reward.

---

### 17 March 2025, 4h

- Refactored the _get_position_relative_to_section function of the [MapLayout](src/map_interaction/map_layout.py) class to be more readable.
- Logger now logs much more information such as the model architecture, and statistics about the run.
- Made a [notebook](src/utils/plot_stats.ipynb) to plot the information contained in the log file.
- Allowed to load a model from a pth file and continue training from there.
- Fixed a stupid error in the epsilon value. We had swapped the comparators, so the agent would not make much random moves early on and only random moves at the end.

---

### 18 March 2025, 3h

- Did some profiling but didn't find anything interesting.
- Fixed a bug we had with the old and current states. They were the same, so the agent was confused. Now surely the agent will be very smart.

---

### 19 March 2025, 3h

- Profiled the time it took for an execution of the on_run_step function to determine a good game speed.
- Created a simpler track for the agent to train in. It is a straight line.
- Optimized the train_step function to run faster, now avoiding moving from GPU to CPU and back.

#### Tried

- Modified the reward to be multiplied by the in game speed. This way, the agent will be rewarded for going faster.
- Added a multiplicative penalty to the reward. If the agent collides with the wall, its reward will be divided by 5, and if it finishes the track, it will be multiplied by 10.
- Changed the next_turn input to the in game speed because there are no turns in the new track.
- Increased the learning rate to 0.005 from 0.001 to try to speed up the learning
- Increased the number of layers to 2 layers of 128 neurons each.
- As the epsilon decreases, the agent's score is becoming worse and worse. It looks like it is not learning anything.

---

### 21 March 2025

- It works on the straight map!
- The problem was that we had a Sigmoid activation function in the output layer. We changed it to a Linear activation function and now it works. This is because with a Sigmoid, there was always a loss that was detected between the prediction and the target
- The agent is now able to finish the track, and get the author time (press forward map)
