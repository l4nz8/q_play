from gym_env import MarioGymAI
#from logging import logger
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
    quiet = False

    pyboy = PyBoy(gamerom_file='gb_ROM/SuperMarioLand.gb',
                  window_type="headless" if quiet else "SDL2",
                  window_scale=3,
                  sound=False,
                  cgb=False,
                  debug=False,
                  game_wrapper=True)
    env = PyBoyGymEnv(pyboy, observation_type='raw', action_type='toggle', simultaneous_actions=False)

    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, shape=84)  # transform MultiDiscreate to Box for framestack
    env = NormalizeObservation(env)  # normalize the values
    env = FrameStack(env, num_stack=4)
   
    mario = MarioGymAI(state_dim=(4, 84, 84))

    episodes = 40
    for e in range(episodes):
        state = env.reset()
        start = time.time()

        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            #logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or start > 400:
                break

        #logger.log_episode()
