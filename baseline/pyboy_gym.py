from pyboy.openai_gym import PyBoyGymEnv
from qnet_interface import MarioAI

class CustomPyBoyGym(PyBoyGymEnv):
    def __init__(self, pyboy, observation_type="tiles", action_type="toggle", simultaneous_actions=False, load_initial_state=False,**kwargs):
        super().__init__(pyboy, observation_type, action_type, simultaneous_actions, **kwargs)
    
        self.load_initial_state=load_initial_state
        self.state_file = "gb_ROM/SuperMarioLand.gb.state"

    def step(self, list_actions):
        """
            Simultanious action implemention
        """
        info = {}
        
        previousGameState = self.aiSettings.GetGameState(self.pyboy)

        if list_actions[0] == self._DO_NOTHING:
            pyboy_done = self.pyboy.tick()
        else:
            # release buttons if not pressed now but were pressed in the past
            for pressedFromBefore in [pressed for pressed in self._button_is_pressed if self._button_is_pressed[pressed] == True]: # get all buttons currently pressed
                if pressedFromBefore not in list_actions:
                    release = self._release_button[pressedFromBefore]
                    self.pyboy.send_input(release)
                    self._button_is_pressed[release] = False

            # press buttons we want to press
            for buttonToPress in list_actions:
                self.pyboy.send_input(buttonToPress)
                self._button_is_pressed[buttonToPress] = True # update status of the button

            pyboy_done = self.pyboy.tick()

        # reward 
        reward = self.aiSettings.GetReward(previousGameState, self.pyboy)

        observation = self._get_observation()

        done = pyboy_done or self.pyboy.game_wrapper().game_over()
        return observation, reward, done, None, info

    def setAISettings(self, aisettings: MarioAI):
        self.aiSettings = aisettings

    def set_world_level(self, world, level):
        """
        Setzt das Spiel auf angegebene Welt- und Level.
        """
        # Sicher stelle, dass PyBoy und das Spiel bereits geladen sind
        if not self.pyboy:
            raise ValueError("PyBoy ist nicht initialisiert oder kein Spiel geladen.")

        for i in range(0x450, 0x461):
            self.pyboy.override_memory_value(0, i, 0x00)

        patch1 = [
            0x3E, # LD A, d8
            (world << 4) | (level & 0x0F), # d8
        ]

        for i, byte in enumerate(patch1):
            self.pyboy.override_memory_value(0, 0x451 + i, byte)

    def reset(self):
        """ Reset (or start) the gym environment throught the game_wrapper """
        if not self._started:
            self.game_wrapper.start_game(**self._kwargs)
            self._started = True
        else:
            self.game_wrapper.reset_game()

        if self.load_initial_state:
            with open(self.state_file, "rb") as f:
                self.pyboy.load_state(f)

        # release buttons if not pressed now but were pressed in the past
        for pressedFromBefore in [pressed for pressed in self._button_is_pressed if self._button_is_pressed[pressed] == True]: # get all buttons currently pressed
            self.pyboy.send_input(self._release_button[pressedFromBefore])
        self.button_is_pressed = {button: False for button in self._buttons} # reset all buttons

        return self._get_observation(), None