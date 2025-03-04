import multiprocessing
import os
from time import sleep
from tminterface.interface import TMInterface

from client import MainClient
import signal

class Worker(multiprocessing.Process):
    def __init__(self, server_id):
        super().__init__()
        self.server_id = server_id
        self.client = MainClient(self.server_id)
        self.iface = TMInterface(f"TMInterface{self.server_id}")

    def signal_handler(self, sig, frame):
        self.iface.execute_command("quit")
        self.iface.close()

    def run(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self.iface.register(self.client)

        while self.iface.running:
            sleep(0)
