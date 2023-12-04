from gym_env import MarioGymAI
from pyboy import PyBoy, WindowEvent
from pyboy_gym import CustomPyBoyGym
from qnet_interface import MarioAI
import torch
import time
from wrapper import SkipFrame, ResizeObservation
from gym.wrappers import FrameStack, NormalizeObservation

if __name__ == '__main__':
    
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()
    episodes = 4000
    frameStack = 4
    gameDimentions = (20, 16)
    quiet = False
    action_types = ["press", "toggle", "all"]
    observation_types = ["raw", "tiles", "compressed", "minimal"]

    # Load emulator
    pyboy = PyBoy(gamerom_file='gb_ROM/SuperMarioLand.gb',
                  window_type="headless" if quiet else "SDL2",
                  #window_scale=3,
                  sound=False,
                  cgb=False,
                  debug=False,
                  game_wrapper=True)
    
    # Load envirament
    ai_interface = MarioAI()
    env = CustomPyBoyGym(pyboy, observation_type=observation_types[0])
    env.setAISettings(ai_interface)  # use this settings
    filteredActions = ai_interface.GetActions()  # get possible actions
    print("Possible actions: ", [[WindowEvent(i).__str__() for i in x] for x in filteredActions])

    # Apply wrappers on env.
    """
    !!!!!Bug!!!!!!
    """
    env = SkipFrame(env, skip=4)
    #env = ResizeObservation(env, shape=gameDimentions)  # transform MultiDiscreate to Box for framestack
    env = NormalizeObservation(env)  # normalize the values
    #env = FrameStack(env, num_stack=frameStack)
    
    # Load AI
    mario = MarioGymAI(state_dim=(frameStack,) + gameDimentions, action_space_dim=len(filteredActions)) #state_dim=(framestack,(window_shape))

    # Setup emulator parameters
    pyboy.set_emulation_speed(0)
    print("Training mode")
    print("Total Episodes: ", episodes)
    #mario.net.train()

    # Training
    for e in range(episodes):
        state, info = env.reset()
        #print(state.shape)
        #exit()
        start = time.time()

        # Play the game!
        while True:

            # Action based on current state
            actionId = mario.act(state)
            action = filteredActions[actionId]

            # Agent performs action
            next_state, reward, done, truncated, info = env.step(action)
            # Remember
            mario.cache(state, next_state, actionId, reward, done)

            # Learn
            q, loss = mario.learn()

            # Update state
            state = next_state

            # Check if end of game
            if done or time.time() - start > 500:
                break

    env.close()