from gym_env import MarioGymAI
from pyboy import PyBoy, WindowEvent
from pyboy_gym import CustomPyBoyGym
import console_ui as ui
from qnet_interface import MarioAI
import torch
import time
from wrapper import SkipFrame, ResizeObservation, GrayPermuteObservation
from gym.wrappers import FrameStack, NormalizeObservation

import datetime
import os
#from logger import MetricLogger
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import argparse

# Argument Parser Setup
parser = argparse.ArgumentParser(description="\U0001F916 Train or Play with Mario AI", 
                                 epilog="""\U0001F4BB To display the training progress, open Tensorboard in your browser
                                 
\U0001F3AE PyBoy emulator controls:
  I: Toggle screen recording (press again to stop)
  O: Take screenshot
  X: Save game state
  Z: Load game state
  P: Pause Game
  Space: Toggle Unlimited FPS on/off
  ESC: Exit the game (Do not use in training mode!!!)
These controls allow direct interaction with the emulator during gameplay.

To display the training progress, open Tensorboard in your browser""", 
formatter_class=argparse.RawTextHelpFormatter)

group = parser.add_mutually_exclusive_group()

# Erweiterung des Argument Parsers
group.add_argument("--world", type=int, default=1, help="Specify the world to play/train in (z.B. 1 für Welt 1) default=1")
group.add_argument("--level", type=int, default=1, help="Specify the lvl to play/train in (z.B. 1 für Karte 1) default=1")

parser.add_argument("-m", "--mode", choices=["train", "play"], default="train", 
                    help="Mode to run the AI: 'train' for training, 'play' to play with a trained model. (default=train)")

parser.add_argument("--headless", action="store_true",
                    help="Activates headless mode to run the emulator without rendering the game screen, which can increase the speed of training.")

parser.add_argument("-ls", "--load-state", action="store_true",
                    help="Load a saved game state from the default save location (gb_ROM/SuperMarioLand.gb.state).")

parser.add_argument("-los", "--load-optimizer-state", action="store_true",
                    help="Load the optimizer and scheduler state from the checkpoint.")

parser.add_argument("-lrs", "--lr_scheduler", choices=["StepLR", "Cyclic"], default="StepLR",
                    help="Select the learning rate scheduler for the Adam optimizer. (default=StepLR)")

parser.add_argument("-exp", "--exploration", type=float, default=None,
                    help="Set a custom exploration rate for the model after loading.")

args = parser.parse_args()

if __name__ == '__main__':
    
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()
    episodes = 40000 
    frameStack = 4
    gameDimentions = (84, 84)
    quiet = args.headless
    action_types = ["press", "toggle", "all"]
    observation_types = ["raw", "tiles", "compressed", "minimal"]

    """
    Logger
    """
    save_dir = "checkpoints"

    if args.mode == "train":
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
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
    env = CustomPyBoyGym(pyboy, observation_type=observation_types[0], load_initial_state=args.load_state)
    env.setAISettings(ai_interface)  # use this settings
    filteredActions = ai_interface.GetActions()  # get possible actions
    print("Possible actions: ", [[WindowEvent(i).__str__() for i in x] for x in filteredActions])

    # Setze Welt und Level
    world = args.world
    level = args.level
    env.set_world_level(world, level)  # Aufruf der neuen Methode

    # Apply wrappers on env.
    env = SkipFrame(env, skip=4)
    env = GrayPermuteObservation(env)
    env = ResizeObservation(env, shape=gameDimentions)  # transform MultiDiscreate to Box for framestack
    env = FrameStack(env, num_stack=frameStack)
    
    # Load AI
    mario = MarioGymAI(state_dim=(frameStack,) + gameDimentions, action_space_dim=len(filteredActions), save_dir=save_dir, args=args)

    # Laden der checkpoints
    available_checkpoints = ui.list_available_models_and_params(save_dir)
    ui.load_model_interactively(available_checkpoints, save_dir, mario, args)

    # Training
    if args.mode == "train":
        # Setup emulator parameters
        pyboy.set_emulation_speed(0)
        print("Training mode activated.")
        print("Total Episodes: ", episodes)
        # Führe den Trainingsmodus aus
        for e in range(episodes):
            state, info = env.reset()
            step = 0
            episode_reward = 0
            episode_loss = []
            episode_q = []
            #print(e)
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

            # save model
            #if (e % 20 == 0) or (e == episodes - 1): # save every 20 episodes and at the end
            #    mario.save()

            #if (e % 20 == 0) or (e == episodes - 1):
            #   logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

    # Playing
    elif args.mode == "play":
        print("Play mode activated.")
        # Führe den Spielmodus aus
        state, info = env.reset()
        total_reward = 0
        while True:
            actionId = mario.act(state, train_mode=False)  # Verwende das Modell, um die Aktion zu wählen
            action = filteredActions[actionId]
            next_state, reward, done, truncated, info = env.step(action)

            total_reward += reward
            
            # Update state
            state = next_state
            if done:
                break
        print(f"Total score: {total_reward}")

    env.close()
    writer.close()
    exit()