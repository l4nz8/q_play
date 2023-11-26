import gym_env
from pyboy import PyBoy, WindowEvent

pyboy = PyBoy(gamerom_file='gb_ROM/SuperMarioLand.gb',sound=False,cgb=True, game_wrapper=True)

actions = [WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B]
relase_arrow = [WindowEvent.RELEASE_ARROW_DOWN,
                WindowEvent.RELEASE_ARROW_LEFT,
                WindowEvent.RELEASE_ARROW_RIGHT,
                WindowEvent.RELEASE_ARROW_UP]
release_button = [WindowEvent.RELEASE_BUTTON_A,
                    WindowEvent.RELEASE_BUTTON_B]

pyboy.set_emulation_speed(1)
mario = pyboy.game_wrapper()
mario.start_game()

pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
while not pyboy.tick():


    pyboy.tick()
    if mario.lives_left == 1:

        print(mario)
        mario.reset_game()


assert mario.lives_left == 2

pyboy.stop()

# exampel code to 
#while not pyboy.tick():
#    pass
#pyboy.stop()