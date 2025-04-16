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

- Refactored the _get_position_relative_to_section function of the ~~[MapLayout]()~~ class to be more readable.
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

### 21 March 2025, 5h

- It works on the straight map!
- The problem was that we had a Sigmoid activation function in the output layer. We changed it to a Linear activation function and now it works. This is because with a Sigmoid, there was always a loss that was detected between the prediction and the target
- The agent is now able to finish the track, and get the author time (press forward map)
- Moving back to the first map to see if the agent can now improve on it
- Added back the next_turn input but kept the in game speed as an input. We now have 5 inputs: the relative position of the car on the road section, the relative yaw, the in game speed, the next turn.
- Removed the bonus for finishing the track, we still need to detect that we are close to the finish for that.
- Started creating an interface to have a better visualization of the training process. Right now, we have the game, the plot of the reward. Some buttons to replace the hotkeys and a slider to change the epsilon value.

---

### 22 March 2025, 7h

- Manual control of epsilon is now possible with the slider. The value is updated in real time.
- Added visualization of the actions taken by the agent. We can see the reward that the agent expects to get from the action.
- Added a button to load a model from a directory. Still need to load the hyperparameters of the model.

---

### 23 March 2025, 6h

- When loading a model, the hyperparameters are now loaded too. The model can be trained from where it was left off. Also, all the previous iterations are copied to the new log file.
- Added a boolean to allow the agent to randomly spawn on the map using states, and made a script to create these states. For now, we will not use it because it should be possible for the agent to still learn something without it which is not really the case now.
- Did a few runs, not very successful yet.
- Tried to tweak the learning rate.
- Fixed a bug in the reward calculation. One term was too large.
- Started to change the way the agent views the map. Now we will use a graph where the turns are the nodes. The agent's position will be based on the edge it is closest to.

---

### 24 March 2025, 6h

- Work in the graph branch. We can now properly get the relative position of the agent on the road section and calculate the reward based on the distance traveled.
- Rescaled the reward to meters. The agent now gets a reward of 1 for every meter in the track's direction traveled.
- Added next turn and relative yaw to the new [AgentPosition](src/map_interaction/agent_position.py) file
- Added a curve to the real time plot to visualize the average reward over the last 200 iterations.

---

### 26 March 2025, 1h

- Implemented a double DQN. We previously had an issue where the reward values would explode out of control. This is probably because we updated the same network that was used to calculate the target. Now we have two networks, one to calculate the target and one to update the weights. The target network is updated every 5 iterations (parameter in the config).
- Added a label to display the largest reward to the real time plot.

---

### 27 March 2025, 3h

- Changed the state to include the acceleration of the car.
- Removed the clamp on the position
- Added a penalty if the agent dies, or stops moving. It is a fixed penalty of -20.

---

### 28 March 2025, 2h

- Added a priority replay buffer. The priority is based on the huber loss between the target and the prediction.
- Changed the target update to 2000 iterations.
- Removed the train_short memory.

---

### 5 April 2025, 2h
- Tried several different discount factors: 0.9, 0.96 and 0.99. The last one seems to be the best, but we don't know why.
- Tried implementing n-step learning. We are launching training to see if it works.
- Made a simple map that is just a series of turns, each separated by the same distance. We will use this map because it is easier for the agent to learn.

---

### 7 April 2025, 2h
- Started implementing PPO instead of DQN. We used the same architecture as the DQN agent.
- Tried training without entropy term, the agent was too greedy and didn't explore enough.
- Added the entropy term to the loss function. There is still a bug in the advantages so the agent could not learn properly.

---

### 8 April 2025, 2h
- Fixed the bug in the advantages.
- Tried training with a horizon of 40 steps, it was terrible.
- Normalized the advantages.

---

### 9 April 2025, 2h
- Tried looking at the PPO implementation to understand what went wrong
- Removed some detach() from the RolloutBuffer
- Added multiple epochs to the train step
- Trained the agent when running with multiple epochs and suddenly it is the best performance we ever had, completely dominating the DQN in terms of speed of learning.

---

### 12 April 2025, 5h
- Merged the PPO and DQN agents, now we can select which one to use when launching the training.
- Allowed to modify the game speed for the DQN agent.
- Fixed a bug when the game was running at high speeds that would cause the interface and the terminal to be out of sync.

---

###  14 April 2025, 7h
- Following Yosh's advice, we implemented IQN. Still not working properly, but we are getting there.
- Maybe we should use curriculum learning to train the agent.
- Optimized priority replay buffer with tensors
- Tried epsilon boltzmann policy but didn't keep it because it was too exploitative for now.
- Added an evaluation mode for the PPO agent

---

### 15 April 2025, 5h
- Finished the IQN implementation.
- Changed the input to the network to be the distance to the next corner, normalized by 10 because the maximum section length is 10 blocks.
- [ ] We should implement a function to determine the largest section because right now we are hardcoding it.
- Fixed a bug in the reward function that would give the agent a negative reward when it was crossing the corner on the exterior side. This was the reason why the agent didn't want to cross the corner when it lander on the exterior side.
- Completely removed any reward when the agent is on the exterior side of the corner. This should help the agent learn to cross the corner on the interior side. It should also avoid suboptimal strategies where the agent would go on the exterior side of the corner to get a reward.
- Started a run for the night

---

### 16 April 2025, 3h30
- This run was incredible! For the first time, the agent completed the map, and we didn't even need to use random spawn or curriculum learning.
- It managed to improve its time from 8:00 at first to 5:00 right now (450h of training).
- Changed the mechanism to save a model, now we save in the same directory, instead of creating a new one each time. This will be easier to manage.
- Added tracking of personal best time.
