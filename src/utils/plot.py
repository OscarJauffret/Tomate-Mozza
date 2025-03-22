import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')

class Plot:
    def __init__(self, parent, plot_size=100, title="Real-time Bar Chart", xlabel="Iteration", ylabel="Value"):
        self.parent = parent
        self.plot_size = plot_size
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        # Data containers
        self.x_data = np.array([], dtype=int)
        self.y_data = np.array([], dtype=float)

        # Setup the figure
        self.fig, self.ax = plt.subplots()
        self.bars = None  # BarContainer to store rectangles

        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.grid(True)
        self.ax.set_axisbelow(True)  # Ensure grid is behind bars

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.iterations = 0

    def add_point(self, y):
        self.iterations += 1

        self.x_data = np.append(self.x_data, self.iterations)
        self.y_data = np.append(self.y_data, y)

        # Keep only the most recent plot_size points
        if len(self.x_data) > self.plot_size:
            self.x_data = self.x_data[-self.plot_size:]
            self.y_data = self.y_data[-self.plot_size:]

        if self.bars is None:
            # Create bar chart once
            self.bars = self.ax.bar(self.x_data, self.y_data, width=1.0, align='center', color='#37504b')
        else:
            # If number of bars changed, recreate
            if len(self.bars) != len(self.y_data):
                self.ax.clear()
                self.ax.set_title(self.title)
                self.ax.set_xlabel(self.xlabel)
                self.ax.set_ylabel(self.ylabel)
                self.ax.grid(True)
                self.ax.set_axisbelow(True)
                self.bars = self.ax.bar(self.x_data, self.y_data, width=1.0, align='center', color='#37504b')
            else:
                # Update bar heights
                for rect, new_height in zip(self.bars, self.y_data):
                    rect.set_height(new_height)

        # Adjust axes
        self.ax.relim()
        self.ax.autoscale_view()

        self.canvas.draw()

    def clear(self):
        self.x_data = np.array([], dtype=int)
        self.y_data = np.array([], dtype=float)
        self.ax.cla()
        self.bars = None
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.grid(True)
        self.ax.set_axisbelow(True)
        self.canvas.draw()

    def close(self):
        plt.close(self.fig)

    def pause(self):
        plt.pause(0.1)