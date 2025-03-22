import tkinter as tk
from tkinter import ttk
from ..utils.plot import Plot
import pygetwindow as gw
import win32gui
import win32con
from src.config import Config
from ..utils.utils import trigger_map_event
from time import sleep

class Interface:
    def __init__(self, choose_map_event, print_state_event, save_model_event, quit_event, epsilon_queue) -> None:        
        self.root = tk.Tk()
        self.root.title("Tomate Mozza")
        self.full_screen = False
        self.game_frame = None
        self.graph_frame = None
        self.button_frame = None
        self.choose_map_event = choose_map_event
        self.print_state_event = print_state_event
        self.save_model_event = save_model_event
        self.quit_event = quit_event

        self.epsilon_scale = None
        self.epsilon_queue = epsilon_queue
        self.epsilon_toggle = tk.IntVar(value=1)
        self.previous_epsilon_value = None
        self.previous_toggle_value = None

        self.game_geometry = (640, 480)
        self.graph_geometry = (640, 480)

        self.create_game_frame()
        self.create_graph_frame()
        self.create_button_frame()
        self.create_epsilon_scale()

        self.root.bind("<F11>", self.toggle_fullscreen)

        self.graph = Plot(parent=self.graph_frame, plot_size=20000, title="Reward", xlabel="Iteration", ylabel="Reward")

        self.embed_trackmania(self.game_frame)

        self.after_id = None
        self.after_epsilon_id = None

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
        self.button_frame.grid(row=2, column=0, padx=10)
        self.load_map_button = ttk.Button(self.button_frame, text="Load the map", command=lambda: trigger_map_event(self.choose_map_event))
        self.load_map_button.grid(row=0, column=0, padx=5, sticky="nsew")
    
        self.print_state_button = ttk.Button(self.button_frame, text="Print the state", command=lambda: self.print_state_event.set())
        self.print_state_button.grid(row=0, column=1, padx=5,sticky="nsew")

        self.save_model_button = ttk.Button(self.button_frame, text="Save the model", command=lambda: self.save_model_event.set())
        self.save_model_button.grid(row=0, column=2, padx=5, sticky="nsew")

        self.quit_button = ttk.Button(self.button_frame, text="Quit", command=self.close_window)
        self.quit_button.grid(row=0, column=3, padx=5, sticky="nsew")

        self.toggle_epsilon_scale = tk.Checkbutton(self.button_frame, text="Disable Manual Epsilon", command=self.toggle_epsilon, 
                                                   variable=self.epsilon_toggle)
        self.toggle_epsilon_scale.grid(row=0, column=4, padx=5, sticky="nsew")

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)  

    def create_epsilon_scale(self):
        """Create the scale"""
        self.epsilon_scale = tk.Scale(self.root, from_=1, to=0, orient="horizontal",  
                                    tickinterval=0.1, length=400, label="Epsilon", 
                                    resolution=0.01 )
        self.epsilon_scale.grid(row=1, column=0, padx=5, sticky="nsew")
        self.epsilon_scale.set(1)

        self.epsilon_scale["state"] = "disabled"   # Initially disabled
    
    def toggle_epsilon(self):
        """Toggle the epsilon value"""
        if self.epsilon_toggle.get() == 0:
            self.epsilon_scale["state"] = "normal"
        else:
            self.epsilon_scale["state"] = "disabled"
        self.send_manual_epsilon()
    
    def send_manual_epsilon(self):
        """Send the manual epsilon value to the client"""
        current_epsilon_value = self.epsilon_scale.get()
        current_epsilon_toggle = self.epsilon_toggle.get()
        if self.epsilon_toggle.get() == 0:
            if current_epsilon_value != self.previous_epsilon_value:
                self.epsilon_queue.put(("Enabled", current_epsilon_value))
                self.previous_epsilon_value = current_epsilon_value
        else:
            if current_epsilon_toggle != self.previous_toggle_value:
                self.epsilon_queue.put(("Disabled", None))
                self.previous_toggle_value = current_epsilon_toggle
        self.after_epsilon_id = self.root.after(50, self.send_manual_epsilon)
    

    def update_graph(self, queue):
            if not queue.empty():
                reward = queue.get()
                self.graph.add_point(reward)

            self.after_id = self.root.after(100, self.update_graph, queue)

    def on_close(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        
        if self.after_epsilon_id:
            self.root.after_cancel(self.after_epsilon_id)

        self.root.quit()
        self.root.destroy()

    def close_window(self):
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

# Run the interface
if __name__ == "__main__":
    app = Interface()
    app.run()
