from gym_env import MarioGymAI
from pyboy import PyBoy, WindowEvent
from pyboy_gym import CustomPyBoyGym
from qnet_interface import MarioAI
import torch
import time
from wrapper import SkipFrame, ResizeObservation
from gym.wrappers import FrameStack, NormalizeObservation

import datetime
import os
#from logger import MetricLogger
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()
    episodes = 4000 
    frameStack = 3
    gameDimentions = (20, 16)
    quiet = False
    action_types = ["press", "toggle", "all"]
    observation_types = ["raw", "tiles", "compressed", "minimal"]

    """
    Logger
    """
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = "checkpoints"
    # TensorBoard writer
    writer = SummaryWriter(f"runs/mario_experiment_{now}")

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
    mario = MarioGymAI(state_dim=(frameStack,) + gameDimentions, action_space_dim=len(filteredActions), save_dir=save_dir) #state_dim=(framestack,(window_shape))
    
    # check for model_checkpoint
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Checkpoint directory not found, create a directory.")

    else:
        checkpoints = [file for file in os.listdir(save_dir) if file.endswith(".chkpt")]
        if checkpoints:
            last_checkpoint = os.path.join(save_dir, checkpoints[-1])
            mario.load_model(last_checkpoint)
            print(f"load checkpoint: {checkpoints[-1]}")
        else:
            print("No checkpoints available, start a new training session.")

    # Assuming MetricLogger is a class or object you're using
    #logger = MetricLogger(save_dir)

    # Setup emulator parameters
    pyboy.set_emulation_speed(0)
    print("Training mode")
    print("Total Episodes: ", episodes)
    #mario.net.train()

    # Training
    for e in range(episodes):
        state, info = env.reset()
        step = 0
        episode_reward = 0
        episode_loss = []
        episode_q = []
        print(e, "LOLOLOLOLOLOLOLOLOLOLOLOLOL")
        #print(state.shape)
        #exit()
        start = time.time()

        # Play the game!
        while True:

            # Action based on current state
            actionId = mario.act(state)
            step +=1
            #print(step)
            action = filteredActions[actionId]

            # Agent performs action
            next_state, reward, done, truncated, info = env.step(action)
            # Remember
            mario.cache(state, next_state, actionId, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            #logger.log_step(reward, loss, q)

            # Updating metrics
            if loss is not None:
                episode_loss.append(loss)
            if q is not None:
                episode_q.append(q)

            episode_reward += reward

            # Update state
            state = next_state

            # Check if end of game
            if done or time.time() - start > 500:
                break
        
        # Am Ende jeder Episode
        avg_loss = sum(episode_loss) / len(episode_loss) if episode_loss else 0
        avg_q = sum(episode_q) / len(episode_q) if episode_q else 0

        # Calculate and log metrics
        avg_reward, avg_length, avg_loss, avg_q = mario.update_moving_averages(episode_reward, step, avg_loss, avg_q)
        writer.add_scalar("Average Reward", avg_reward, e)
        writer.add_scalar("Average Length", avg_length, e)
        writer.add_scalar("Average Loss", avg_loss, e)
        writer.add_scalar("Average Q-Value", avg_q, e)

        #logger.log_episode()
        mario.save()

        #if (e % 20 == 0) or (e == episodes - 1):
        #   logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

    env.close()
    writer.close()
    exit()