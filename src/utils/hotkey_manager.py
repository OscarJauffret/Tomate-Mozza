import keyboard

class HotkeyManager:
    def __init__(self):
        self.hotkeys = {}  # Stores hotkeys and their functions
        self.enabled = True  # Indicates whether hotkeys are active

    def add_hotkey(self, key, function, description=""):
        """
        Registers a new hotkey.
        :param key: The key to press
        :param function: The function to call when the hotkey is pressed
        :param description: A description of what the hotkey does
        :return: None
        """
        self.hotkeys[key] = {"function": function, "description": description}
        keyboard.add_hotkey(key, self.execute_hotkey, args=[key])

    def execute_hotkey(self, key):
        """
        Executes the function associated with a hotkey.
        :param key: The key that was pressed
        :return: None
        """
        if (self.enabled or key == 'f7') and key in self.hotkeys:
            self.hotkeys[key]["function"]()

    def disable_hotkeys(self):
        """Disables all hotkeys."""
        self.enabled = False

    def enable_hotkeys(self):
        """Re-enables all hotkeys."""
        self.enabled = True

    def toggle_hotkeys(self):
        """Toggles hotkeys on or off."""
        self.enabled = not self.enabled
        print(f"Hotkeys {'enabled' if self.enabled else 'disabled'}")

    def print_hotkeys(self):
        """Displays the list of registered hotkeys."""
        for key, data in self.hotkeys.items():
            print(f"Press '{key}' to {data['description']}")
        print("Press 'CTRL+C' to quit")
