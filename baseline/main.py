from gym_env import MarioGymAI
from pyboy import PyBoy, WindowEvent
from pyboy_gym import CustomPyBoyGym
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
parser = argparse.ArgumentParser(description="Train or Play with Mario AI")
group = parser.add_mutually_exclusive_group()

# Erweiterung des Argument Parsers
group.add_argument("--world", type=int, default=1, help="Specify the world to play/train in (z.B. 1 für Welt 1) default=1")
group.add_argument("--level", type=int, default=1, help="Specify the lvl to play/train in (z.B. 1 für Karte 1) default=1")

parser.add_argument("-m", "--mode", choices=["train", "play"], default="train", 
                    help="Mode to run the AI: 'train' for training, 'play' to play with a trained model. (default=train)")

parser.add_argument("-exp", "--exploration", type=float, default=None,
                    help="Set a custom exploration rate for the model after loading. Useful for transfer learning.")

args = parser.parse_args()

def list_available_models_and_params(save_dir):
    if not os.path.exists(save_dir):
        print("Checkpoint directory not found, creating a directory.")
        os.makedirs(save_dir)
        return []
    
    checkpoints = [file for file in os.listdir(save_dir) if file.endswith(".chkpt")]
    checkpoints.sort()  # sortiert die Dateien
    if not checkpoints:
        print("No checkpoints available, start a new training session.")
        return []

    print("Available model versions and their parameters:")
    for idx, model in enumerate(checkpoints, start=1):
        checkpoint_path = os.path.join(save_dir, model)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            exploration_rate = checkpoint.get('exploration_rate', 'Not available')
            print(f"{idx}: {model}, Exploration Rate: {exploration_rate}")
        except Exception as e:
            print(f"Could not read {model}: {e}")
    return checkpoints

def load_model_interactively(available_checkpoints, save_dir, mario):
    if len(available_checkpoints) == 0:
        return
    
    model_version = None
    if len(available_checkpoints) > 1:
        version_input = input("Please enter the version number to load (or press Enter to use the latest): ")
        if version_input:
            try:
                model_version = int(version_input)
                if model_version < 1 or model_version > len(available_checkpoints):
                    raise ValueError("Invalid version number selected.")
            except ValueError:
                print("Invalid input. Exiting.")
                exit()

    selected_checkpoint = available_checkpoints[-1 if model_version is None else model_version - 1]
    checkpoint_path = os.path.join(save_dir, selected_checkpoint)
    print(f"Loading model: {selected_checkpoint}")
    # Modell laden mit mario.load_model(checkpoint_path)
    mario.load_model(checkpoint_path)

    # Anpassen der Exploration Rate, falls spezifiziert
    if args.exploration is not None:
        mario.exploration_rate = args.exploration
        print(f"Exploration Rate set to {args.exploration} for Transfer Learning.")

if __name__ == '__main__':
    
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()
    episodes = 40000 
    frameStack = 4
    gameDimentions = (84, 84)
    quiet = False
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
    env = CustomPyBoyGym(pyboy, observation_type=observation_types[0])
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
    mario = MarioGymAI(state_dim=(frameStack,) + gameDimentions, action_space_dim=len(filteredActions), save_dir=save_dir)

    # Laden der checkpoints
    available_checkpoints = list_available_models_and_params(save_dir)
    load_model_interactively(available_checkpoints, save_dir, mario)

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
            print(e)
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
            #mario.save()

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