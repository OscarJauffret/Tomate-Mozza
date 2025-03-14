import matplotlib.pyplot as plt
import numpy as np
import threading
import queue
import time
import matplotlib
matplotlib.use('TkAgg')


class ThreadedPlot:
    def __init__(self, title, x_label, y_label, max_points=100, update_interval=100):
        """
        Creates a matplotlib plot that runs in a separate thread

        Args:
            max_points (int): Maximum number of points to keep in history
            update_interval (int): Update interval in ms
        """
        self.max_points = max_points
        self.update_interval = update_interval

        self.title = title
        self.x_label = x_label
        self.y_label = y_label

        # Queue for communication between threads
        self.data_queue = queue.Queue()

        # Start the plot thread
        self.plotting_thread = threading.Thread(target=self._run_plot_thread)
        self.plotting_thread.daemon = True  # The thread will close when the main program ends
        self.plotting_thread.start()

        # Flag to stop cleanly
        self.running = True

    def _run_plot_thread(self):
        """Function executed in the separate thread"""
        # Data
        iterations = []
        scores = []

        # Create the figure and axes
        plt.ion()  # Interactive mode
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [], 'b-', lw=2)

        # Configure the axes
        ax.set_title(self.title)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.grid(True)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)

        # Main loop of the plot thread
        while self.running:
            # Retrieve all available data from the queue
            updated = False
            while not self.data_queue.empty():
                iteration, score = self.data_queue.get()
                iterations.append(iteration)
                scores.append(score)
                updated = True

                # Limit the number of points
                if len(iterations) > self.max_points:
                    iterations = iterations[-self.max_points:]
                    scores = scores[-self.max_points:]

            # Update the plot if new data has arrived
            if updated and iterations:
                # Update the data
                line.set_data(iterations, scores)

                # Adjust the axes
                ax.set_xlim(max(0, min(iterations)), max(iterations) + 1)
                ax.set_ylim(max(0, min(scores) - 0.1), max(scores) + 0.1)

                # Redraw
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            # Small pause to avoid overloading the CPU
            time.sleep(self.update_interval / 1000)

        # Close the figure at the end
        plt.close(fig)

    def add_point(self, iteration, score):
        """
        Adds a point to the plot (can be called from the main thread)

        Args:
            iteration (int): Iteration number
            score (float): AI score
        """
        self.data_queue.put((iteration, score))

    def close(self):
        """Cleanly stops the plot thread"""
        self.running = False
        if self.plotting_thread.is_alive():
            self.plotting_thread.join(timeout=1.0)  # Wait for the thread to finish


# Example usage
if __name__ == "__main__":
    # Create the plot instance
    plot = ThreadedPlot("AI Training Progress", "Iteration", "Score", max_points=50, update_interval=100)

    # Simulate AI training in the main thread
    try:
        for i in range(100):
            # Simulate a training step
            score = min(0.95, 0.5 + i * 0.005 + np.random.normal(0, 0.05))

            # Add the point to the plot (running in another thread)
            plot.add_point(i, score)

            # Your AI continues its execution normally here
            time.sleep(0.2)  # Simulate work


    finally:
        # Ensure to cleanly close the plot thread
        plot.close()