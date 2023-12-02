from gym_env import MarioGymAI
from wrapper import SkipFrame, ResizeObservation
from pyboy import PyBoy
from pyboy.openai_gym import PyBoyGymEnv
import torch
import time
from gym.wrappers import FrameStack, NormalizeObservation

if __name__ == '__main__':
    
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()
    episodes = 40000
    quiet = False
    train = False
    action_types = ["press", "toggle", "all"]
    observation_types = ["raw", "tiles", "compressed", "minimal"]

    # Load emulator
    pyboy = PyBoy(gamerom_file='gb_ROM/SuperMarioLand.gb',
                  window_type="headless" if quiet else "SDL2",
                  window_scale=3,
                  sound=False,
                  cgb=False,
                  debug=False,
                  game_wrapper=True)
    
    # Load envirament
    """
    !!!!!Bug!!!!!!
    """
    env = PyBoyGymEnv(pyboy, observation_type=observation_types[1], action_type=action_types[0], simultaneous_actions=False)

    # Apply wrappers on env.
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, shape=84)  # transform MultiDiscreate to Box for framestack
    env = NormalizeObservation(env)  # normalize the values
    env = FrameStack(env, num_stack=4)
    
    # Load AI
    """apply variables/values !!!!Bug!!!"""
    mario = MarioGymAI(state_dim=(4, 84, 84)) #state_dim=(framestack,(window_shape))

    # Setup emulator parameters
    pyboy.set_emulation_speed(1)
    
    # Training
    for e in range(episodes):
        state = env.reset()
        start = time.time()

        # Play the game!
        while True:

            # Action based on current state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Update state
            state = next_state

            # Check if end of game
            if done or start > 400:
                break
    env.close()