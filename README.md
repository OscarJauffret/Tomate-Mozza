# Horizon

![Language](https://img.shields.io/badge/language-Python-blue)
![Status](https://img.shields.io/badge/status-InProgress-yellow)
![GitHub Stars](https://img.shields.io/github/stars/OscarJauffret/Tomate-Mozza)
![GitHub Contributors](https://img.shields.io/github/contributors/OscarJauffret/Tomate-Mozza)
![GitHub Last Commit](https://img.shields.io/github/last-commit/OscarJauffret/Tomate-Mozza)
![GitHub Top Language](https://img.shields.io/github/languages/top/OscarJauffret/Tomate-Mozza)

This is a reinforcement learning project that aims to teach an agent to play the game Trackmania Nations Forever.
TMInterface is used to interact with the game.

> [!Note]
> This is a university project, it is still under development.

## Requirements

- [Trackamania Nations Forever (Steam, it's free)](https://store.steampowered.com/app/11020/TrackMania_Nations_Forever/)
- [TMInterface 1.4.3](https://github.com/donadigo/TMInterfaceClientPython)
- [TMInfinity](https://archive.org/download/tminfinity-1.3.0.1)
- [Python packages](requirements.txt)

## Setup

1. Place the contents of the TMInterface/TMInfinity folders in the game directory of TMNF. The game directory is usually located in `..\Steam\steamapps\common\TrackMania Nations Forever`.

2. Edit the config of TMInterface (usually located in `C:\Users\User\Documents\TMInterface\config.txt`) and write `load_infinity`

3. Create a TMNF account and login to it. Then set the autologin variable of TMInterface to your username by typing `set autologin <username>` in the TMInterface console.

4. Setup .env file based on the `.env.template` file. 

5. Install the required packages by running `pip install -r requirements.txt` in the terminal.

6. Create a directory in the [maps](maps) directory and create a folder with a name that allows you to recognize the map.

7. Go to [config.py](src/config.py) and verify that the Config.Paths.MAP variable is set to the same as the filename you chose in step 6.

8. > [!Caution] the following step only works on Linux (install a WSL if you are on Windows)
    - Install pygbx by running `pip install pygbx`
    - Run the following command to generate the layout of the map:
    ```bash
    python3 make_layout.py > layout.txt
    ```
    - Copy the layout.txt file to the maps folder you created in step 6.

9. Execute the [map_graph](src/map_interaction/map_graph.py) script

10. If you are still here you are ready to go! You can now run the [main.py](src/main.py) with the following command:
   ```python
    python -m src.main --alg [DQN|PPO] --name <name>
   ```  
   
   
