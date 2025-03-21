import tkinter as tk
from tkinter import ttk
from ..utils.plot import Plot
import pygetwindow as gw
import win32gui
import win32con
from src.config import Config

class Interface:
    def __init__(self):        
        self.root = tk.Tk()
        self.root.title("Tomate Mozza")
        self.full_screen = False
        self.game_frame = None
        self.graph_frame = None

        self.game_geometry = (640, 480)
        self.graph_geometry = (640, 480)

        self.create_game_frame()
        self.create_graph_frame()

        self.root.bind("<F11>", self.toggle_fullscreen)

        # Crée le graphique
        self.graph = Plot(parent=self.graph_frame, plot_size=20000, title="Reward", xlabel="Iteration", ylabel="Reward")

        # Intègre Trackmania dans la fenêtre
        self.embed_trackmania(self.game_frame)

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
        self.game_frame.pack(side="left")

    def create_graph_frame(self):
        """Create the graph frame"""
        self.graph_frame = ttk.Frame(self.root, width=self.graph_geometry[0], height=self.graph_geometry[1])
        self.graph_frame.pack(side="right")

    def update_graph(self, queue):
            if not queue.empty():
                reward = queue.get()
                self.graph.add_point(reward)

            self.root.after(100, self.update_graph, queue)
            
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
        else:
            print(f"Aucune fenêtre trouvée avec le titre: {Config.Game.WINDOW_NAME}")

# Run the interface
if __name__ == "__main__":
    app = Interface()
    app.run()
