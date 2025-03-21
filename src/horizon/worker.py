import multiprocessing
import signal
from time import sleep
from tminterface.interface import TMInterface
from .client import HorizonClient

class Worker(multiprocessing.Process):
    def __init__(self, server_id, choose_map_event, print_state_event, save_model_event, quit_event, queue, model_path=None, init_iterations=0):
        super().__init__()
        self.server_id = server_id
        self.client = HorizonClient(self.server_id, queue, model_path, init_iterations)
        self.iface = TMInterface(f"TMInterface{self.server_id}")
        self.choose_map_event = choose_map_event
        self.print_state_event = print_state_event
        self.save_model_event = save_model_event
        self.quit_event = quit_event

    def close_signal_handler(self, sig, frame):
        self.iface.execute_command("quit")
        self.iface.close()

    def run(self):
        signal.signal(signal.SIGINT, self.close_signal_handler)
        signal.signal(signal.SIGTERM, self.close_signal_handler)
        self.iface.register(self.client)

        while self.iface.running:
            if self.choose_map_event.is_set():
                self.client.launch_map(self.iface)
                self.choose_map_event.clear()

            if self.print_state_event.is_set():
                print(self.client)
                self.print_state_event.clear()

            if self.save_model_event.is_set():
                self.client.save_model()
                self.save_model_event.clear()

            if self.quit_event.is_set():
                self.close_signal_handler(None, None)
                self.quit_event.clear()

            sleep(0)
