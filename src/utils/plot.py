import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

class Plot:
    def __init__(self, plot_size=100, title="Real-time Plot", xlabel="Time", ylabel="Value"):
        self.plot_size = plot_size
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        
        # Initialize data containers
        self.x_data = np.array([])
        self.y_data = np.array([])
        
        # Setup the figure and axis
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-')
        
        # Configure plot appearance
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.grid(True)
        
        # Display the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Iterations
        self.iterations = 0
    
    def add_point(self, y):
        """Add a new point to the plot and update display"""
        # Add the new data point
        self.iterations += 1
        self.x_data = np.append(self.x_data, self.iterations)
        self.y_data = np.append(self.y_data, y)
        
        # Keep only the most recent plot_size points
        if len(self.x_data) > self.plot_size:
            self.x_data = self.x_data[-self.plot_size:]
            self.y_data = self.y_data[-self.plot_size:]
        
        # Update the plot data
        self.line.set_data(self.x_data, self.y_data)
        
        # Adjust axis limits if needed
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def clear(self):
        """Clear all data points"""
        self.x_data = np.array([])
        self.y_data = np.array([])
        self.line.set_data([], [])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        """Close the plot window"""
        plt.close(self.fig)

    def pause(self):
        """Pause the plot"""
        plt.pause(0.1)
