import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import os

from src.app.plot import Plot
import pygetwindow as gw
import win32gui
import win32con
from src.config import Config
from ..utils.utils import trigger_map_event
from time import sleep
from .action_keys import ActionKeys

class Interface:
    def __init__(self, choose_map_event, print_state_event, load_model_event, save_model_event, quit_event, shared_dict) -> None:
        self.root = tk.Tk()
        self.root.title("Tomate Mozza")
        self.full_screen = False
        self.best_reward = 0

        self.game_frame = None
        self.graph_frame = None
        self.button_frame = None
        self.action_keys: ActionKeys = None
        self.best_reward_label = None

        self.choose_map_event = choose_map_event
        self.print_state_event = print_state_event
        self.load_model_event = load_model_event
        self.save_model_event = save_model_event
        self.quit_event = quit_event
        self.shared_dict = shared_dict

        self.epsilon_scale = None
    
        self.epsilon_toggle = tk.IntVar(value=0)

        self.game_geometry = (640, 480)
        self.graph_geometry = (640, 480)

        self.create_game_frame()
        self.create_graph_frame()
        self.create_button_frame()
        self.create_epsilon_scale()
        self.create_actions_squares()
        self.create_reward_label()

        self.root.bind("<F11>", self.toggle_fullscreen)

        self.graph = Plot(parent=self.graph_frame, plot_size=200, title="Reward", xlabel="Iteration", ylabel="Reward")

        self.embed_trackmania(self.game_frame)

        self.after_id = None

        self.root.protocol("WM_DELETE_WINDOW", lambda: self.close_window)

    def run(self):
        """Run the main loop"""
        self.root.mainloop()

    def toggle_fullscreen(self, event=None):
        """Toggle fullscreen mode"""
        self.full_screen = not self.full_screen
        self.root.attributes("-fullscreen", self.full_screen)
        return "break"

    def create_game_frame(self):
        """Create the game frame"""
        self.game_frame = tk.Canvas(self.root, width=self.game_geometry[0], height=self.game_geometry[1])
        self.game_frame.grid(row=0, column=0, padx=10, pady=10)

    def create_graph_frame(self):
        """Create the graph frame"""
        self.graph_frame = ttk.Frame(self.root, width=self.graph_geometry[0], height=self.graph_geometry[1])
        self.graph_frame.grid(row=0, column=1, padx=10, pady=10)

    def create_button_frame(self):
        """Create the button frame"""
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.grid(row=2, column=0, padx=10, sticky="n")
        self.load_map_button = ttk.Button(self.button_frame, text="Load the map", command=self.load_map)
        self.load_map_button.grid(row=0, column=0, padx=5, sticky="nsew")
    
        self.print_state_button = ttk.Button(self.button_frame, text="Print the state", command=self.print_state_event.set)
        self.print_state_button.grid(row=0, column=1, padx=5,sticky="nsew")

        self.load_model_button = ttk.Button(self.button_frame, text="Load a model", command=self.load_model)
        self.load_model_button.grid(row=0, column=2, padx=5, sticky="nsew")

        self.save_model_button = ttk.Button(self.button_frame, text="Save the model", command=self.save_model_event.set)
        self.save_model_button.grid(row=0, column=3, padx=5, sticky="nsew")

        self.quit_button = ttk.Button(self.button_frame, text="Quit", command=self.close_window)
        self.quit_button.grid(row=0, column=4, padx=5, sticky="nsew")

        self.toggle_epsilon_scale = tk.Checkbutton(self.button_frame, text="Manual Epsilon", command=self.send_manual_epsilon, 
                                                   variable=self.epsilon_toggle)
        self.toggle_epsilon_scale.grid(row=0, column=5, padx=5, sticky="nsew")

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def create_epsilon_scale(self):
        """Create the scale"""
        self.epsilon_scale = tk.Scale(self.root, from_=1, to=0, orient="horizontal",
                                    tickinterval=0.1, length=400, label="Epsilon",
                                    resolution=0.01, command=self.send_manual_epsilon)
        self.epsilon_scale.grid(row=1, column=0, padx=5, sticky="nsew")

    def create_reward_label(self):
        """Create the reward label"""
        self.best_reward_label = ttk.Label(self.root, text=f"Best Reward: {self.best_reward}", font=("Arial", 20))
        self.best_reward_label.grid(row=1, column=1, padx=50, sticky="nsew")

    def create_actions_squares(self):
        self.action_keys = ActionKeys(self.root, 2, 1, key_size=40, padding=3, margin=10)

    def send_manual_epsilon(self, new_epsilon_value=None):
        """Send the manual epsilon value to the client"""
        if self.epsilon_toggle.get() == 1:  
            self.shared_dict["epsilon"]["value"] = self.epsilon_scale.get()
            self.shared_dict["epsilon"]["manual"] = True
        else:
            self.shared_dict["epsilon"]["manual"] = False

    def update_interface(self):
        if not self.shared_dict["reward"].empty():
            reward = self.shared_dict["reward"].get()
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_reward_label["text"] = f"Best Reward: {self.best_reward:.2f}"
            if not self.shared_dict["epsilon"]["manual"]:
                self.graph.add_point(reward)

        if not self.shared_dict["epsilon"]["manual"]:
            self.epsilon_scale.set(self.shared_dict["epsilon"]["value"])

        self.action_keys.update_keys(self.shared_dict["q_values"])
        self.after_id = self.root.after(100, self.update_interface)

    def load_map(self):
        self.load_model_button["state"] = "disabled"
        trigger_map_event(self.choose_map_event)

    def load_model(self):
        models = [model for model in os.listdir(Config.Paths.MODELS_PATH) if os.path.isdir(os.path.join(Config.Paths.MODELS_PATH, model))]

        # Pop-up window
        top = tk.Toplevel()
        top.title("Load a model")
        top.geometry("200x250")
        top.resizable(False, False)
        top.transient(self.root)
        top.grab_set()

        frame = ttk.Frame(top)
        frame.pack(expand=True, fill="both")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        label = ttk.Label(frame, text="Choose a model")
        label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        listbox = tk.Listbox(frame, selectmode="single")
        for model in models:
            listbox.insert(tk.END, model)
        listbox.insert(tk.END, "New model")
        listbox.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Validate button
        button = tk.Button(frame, text="Load", command=lambda: self.load_model_from_listbox(listbox, top))
        button.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

        top.protocol("WM_DELETE_WINDOW", top.destroy)
        self.root.wait_window(top)

    def load_model_from_listbox(self, listbox, top):
        selected_model = listbox.get(listbox.curselection())
        if selected_model == "New model":
            self.load_model_event.set()
        else:
            self.shared_dict["model_path"].put(os.path.join(Config.Paths.MODELS_PATH, selected_model))
            self.load_model_event.set()
        top.destroy()

    def on_close(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        
        self.root.quit()
        self.root.destroy()

    def close_window(self):
        """Close the window"""
        response = messagebox.askyesnocancel("Save model", "Do you want to save the model before quitting?")
    
        if response is None:
            return 
    
        if response:
            self.save_model_event.set()
            sleep(2)
    
        self.quit_event.set()
        self.on_close()

    def embed_trackmania(self, frame):
        windows = gw.getWindowsWithTitle(Config.Game.WINDOW_NAME)

        if windows:
            window = windows[0]
            hwnd = window._hWnd

            win32gui.SetParent(hwnd, frame.winfo_id())

            width, height = frame.winfo_width(), frame.winfo_height()

            win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, win32con.WS_VISIBLE)
            win32gui.MoveWindow(hwnd, 0, 0, width, height, True)
            if width != self.game_geometry[0] or height != self.game_geometry[1]:
                win32gui.MoveWindow(hwnd, 0, 0, self.game_geometry[0], self.game_geometry[1], True)
            win32gui.SetForegroundWindow(hwnd)
        else:
            print(f"Aucune fenêtre trouvée avec le titre: {Config.Game.WINDOW_NAME}")
