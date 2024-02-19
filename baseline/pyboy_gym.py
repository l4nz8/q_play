from pyboy.openai_gym import PyBoyGymEnv
from qnet_interface import MarioAI


class CustomPyBoyGym(PyBoyGymEnv):
    def __init__(
        self,
        pyboy,
        observation_type="tiles",
        action_type="toggle",
        simultaneous_actions=False,
        load_initial_state=False,
        **kwargs
    ):
        """
        Extends the PyBoyGymEnv for custom behavior with Super Mario Land

        Args:
        - pyboy: The PyBoy emulator instance
        - observation_type: Type of observation to use (e.g. 'tiles')
        - action_type: Type of action to use (e.g. 'toggle' for button presses)
        - simultaneous_actions: If True, allows simultaneous button presses
        - load_initial_state: If True, loads the game from a predefined state
        """
        super().__init__(
            pyboy, observation_type, action_type, simultaneous_actions, **kwargs
        )

        self.load_initial_state = load_initial_state
        self.state_file = (
            "gb_ROM/SuperMarioLand.gb.state"  # Path to the game state file
        )

    def step(self, list_actions):
        """
        Executes Simultanious action in the emulator and returns the results.

        Args:
        - list_actions: A list of actions to execute simultaneously.

        Returns:
        - observation: The next state observed after executing the actions.
        - reward: The reward obtained after executing the actions.
        - done: Whether the game is over.
        - info: Additional info (empty dictionary in this case).
        """
        info = {}
        previousGameState = self.aiSettings.GetGameState(self.pyboy)

        # Process no action differently
        if list_actions[0] == self._DO_NOTHING:
            pyboy_done = self.pyboy.tick()
        else:
            # Release buttons not pressed in the current step but were pressed before
            for pressedFromBefore in [
                pressed
                for pressed in self._button_is_pressed
                if self._button_is_pressed[pressed] == True
            ]:  # get all buttons currently pressed
                if pressedFromBefore not in list_actions:
                    release = self._release_button[pressedFromBefore]
                    self.pyboy.send_input(release)
                    self._button_is_pressed[release] = False

            # Press new buttons
            for buttonToPress in list_actions:
                self.pyboy.send_input(buttonToPress)
                self._button_is_pressed[
                    buttonToPress
                ] = True  # update status of the button

            pyboy_done = self.pyboy.tick()

        # Calculate reward based on the game state change
        reward = self.aiSettings.GetReward(previousGameState, self.pyboy)
        observation = self._get_observation()
        done = pyboy_done or self.pyboy.game_wrapper().game_over()

        return observation, reward, done, None, info

    def setAISettings(self, aisettings: MarioAI):
        """
        Configures AI settings for interaction with the game environment

        Args:
        - aiSettings: An instance of MarioAI for reward calculation and action definitions.
        """
        self.aiSettings = aisettings

    def set_world_level(self, world, level):
        """
        Sets the game to a specific world and level by patching memory.

        Args:
        - world: The world number to set.
        - level: The level number within the world to set.
        """
        if not self.pyboy:
            raise ValueError("PyBoy ist nicht initialisiert oder kein Spiel geladen.")

        # Initialize memory patching for world and level selection
        for i in range(0x450, 0x461):
            self.pyboy.override_memory_value(0, i, 0x00)

        patch1 = [
            0x3E,  # LD A, d8
            (world << 4) | (level & 0x0F),  # d8
        ]
        for i, byte in enumerate(patch1):
            self.pyboy.override_memory_value(0, 0x451 + i, byte)

    def reset(self):
        """
        Resets or starts the gym environment through the game wrapper

        Returns:
        - The initial observation after resetting
        """
        if not self._started:
            self.game_wrapper.start_game(**self._kwargs)
            self._started = True
        else:
            self.game_wrapper.reset_game()

        # Load the initial game state if specified
        if self.load_initial_state:
            with open(self.state_file, "rb") as f:
                self.pyboy.load_state(f)

        # Ensure all buttons are released at the start
        for pressedFromBefore in [
            pressed
            for pressed in self._button_is_pressed
            if self._button_is_pressed[pressed] == True
        ]:  # Get all buttons currently pressed
            self.pyboy.send_input(self._release_button[pressedFromBefore])
        self.button_is_pressed = {
            button: False for button in self._buttons
        }  # Reset all buttons

        return self._get_observation(), None
