from gym_env import MarioGymAI
from pyboy import PyBoy, WindowEvent
from pyboy_gym import CustomPyBoyGym
import console_ui as ui
from qnet_interface import MarioAI
import torch
import time
from wrapper import SkipFrame, ResizeObservation, GrayPermuteObservation
from gym.wrappers import FrameStack
import datetime
from torch.utils.tensorboard import SummaryWriter # Logging training metrics
import argparse

# Setup argument parser for command-line options
parser = argparse.ArgumentParser(description="\U0001F916 Train or Play with Mario AI", 
                                 epilog="""\U0001F4BB To display the training progress, open Tensorboard in your browser
                                 
\U0001F3AE PyBoy emulator controls:

  These controls allow direct interaction with the emulator during gameplay.
    I: Toggle screen recording (press again to stop)
    O: Take screenshot
    Z: Save game state
    X: Load game state
    P: Pause Game
    Space: Toggle Unlimited FPS on/off
    ESC: Exit the game (Do not use in training mode!!!)
""", 
formatter_class=argparse.RawTextHelpFormatter)

group = parser.add_mutually_exclusive_group()

# Add arguments for world and level selection
group.add_argument("--world", type=int, default=1, help="Choose the game world to play or train in {e.g. 1 for world 1} (default=1)")
group.add_argument("--level", type=int, default=1, help="Choose the game level to play or train in {e.g. 1 for Level 1} (default=1)")

# Add arguments for mode selection, headless operation, and model state loading
parser.add_argument("-m", "--mode", choices=["train", "play"], default="train", 
                    help="Mode to run the AI: 'train' to train the AI, or 'play' to use a trained model. (default=train)")

parser.add_argument("--headless", action="store_true",
                    help="Enable headless mode to speed up training by not rendering the game screen.")

parser.add_argument("-ls", "--load-state", action="store_true",
                    help="Load a previously saved game state from the default save location (gb_ROM/SuperMarioLand.gb.state).")

parser.add_argument("-los", "--load-optimizer-state", action="store_true",
                    help="Load saved optimizer and scheduler states from a checkpoint for continued training.")

parser.add_argument("-lrs", "--lr_scheduler", choices=["StepLR", "Cyclic"], default="StepLR",
                    help="Choose the learning rate scheduler type for optimization. (default=StepLR)")

parser.add_argument("-exp", "--exploration", type=float, default=None,
                    help="Set a custom exploration rate for the model, overriding the given or default one.")

parser.add_argument("--debug", action="store_true",
                    help="Activate debug mode for verbose output and additional information for troubleshooting.")

args = parser.parse_args()

if __name__ == '__main__':
    
    print("\nMARIO DRL-AGENT (Double Deep Q-Network | Emulator: PyBoy | Game: Super Mario Land):\
          \n\nINFO\tInitialize project...")
    use_cuda = torch.cuda.is_available() # Check for CUDA (GPU) availability for PyTorch
    print(f"Using CUDA: {use_cuda}")
    episodes = 40000 
    frameStack = 4
    gameDimentions = (84, 84)
    quiet = args.headless
    action_types = ["press", "toggle", "all"]
    observation_types = ["raw", "tiles", "compressed", "minimal"]

    """
    Logger setup for training
    """
    save_dir = "checkpoints"
    if args.mode == "train":
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        writer = SummaryWriter(f"runs/mario_experiment_{now}") # TensorBoard writer

    # Load PyBoy emulator with Super Mario Land ROM
    pyboy = PyBoy(gamerom_file='gb_ROM/SuperMarioLand.gb',
                  window_type="headless" if quiet else "SDL2",
                  sound=False,
                  cgb=False,
                  debug=False,
                  game_wrapper=True)
    
    """
    Load enviroment
    """
    # Initialize AI interface for the game
    ai_interface = MarioAI()
    # Create the game environment with the PyBoy emulator and custom settings
    env = CustomPyBoyGym(pyboy, observation_type=observation_types[0], load_initial_state=args.load_state)
    env.setAISettings(ai_interface)  # Apply AI settings to the environment
    filteredActions = ai_interface.GetActions()  # Get possible actions
    print("Possible actions: ", [[WindowEvent(i).__str__() for i in x] for x in filteredActions])

    # Set specified world and level for the game
    world = args.world
    level = args.level
    env.set_world_level(world, level)

    """
    Observation processing
    """
    env = SkipFrame(env, skip=4) # Skip frames to reduce the temporal resolution
    env = GrayPermuteObservation(env) # Convert observations to grayscale
    env = ResizeObservation(env, shape=gameDimentions) # Resize observations to the Box for framestack
    env = FrameStack(env, num_stack=frameStack) # Stack frames to create a temporal dimension
    
    # Load AI agent
    mario = MarioGymAI(state_dim=(frameStack,) + gameDimentions, action_space_dim=len(filteredActions), save_dir=save_dir, args=args)

    # Load model checkpoints if available
    available_checkpoints = ui.list_available_models_and_params(save_dir)
    ui.load_model_interactively(available_checkpoints, save_dir, mario, args)

    # Train
    if args.mode == "train":
        pyboy.set_emulation_speed(0) # Set emulator speed to the fastest
        print("Training mode activated.")
        print("Total Episodes: ", episodes)
        # Training loop over the number of episodes
        for e in range(episodes):
            state, info = env.reset() # Initialize environment
            step = 0
            episode_reward = 0
            episode_loss = []
            episode_q = []
            start = time.time() # Record time of the episode
            
            # Main episode loop
            while True:
                actionId = mario.act(state) # Decide action based on current state
                if args.debug == True:
                    ai_interface.PrintGameState(pyboy)
                step +=1
                action = filteredActions[actionId] # Retrieve action from action ID
                # Execute action and observe the new state and reward
                next_state, reward, done, truncated, info = env.step(action)
                mario.cache(state, next_state, actionId, reward, done) # Store the experience
                q, loss = mario.learn() # Learn
                
                # Update metrics
                if loss is not None:
                    episode_loss.append(loss)
                if q is not None:
                    episode_q.append(q)
                episode_reward += reward
                state = next_state # Update state
                
                # Check if end of game or exceeded max time
                if done or time.time() - start > 500:
                    break
            
            # Calculate average metrics for the episode
            avg_loss = sum(episode_loss) / len(episode_loss) if episode_loss else 0
            avg_q = sum(episode_q) / len(episode_q) if episode_q else 0

            # Update and log moving averages of the metrics
            avg_reward, avg_length, avg_loss, avg_q = mario.update_moving_averages(episode_reward, step, avg_loss, avg_q)
            writer.add_scalar("Average Reward", avg_reward, e)
            writer.add_scalar("Average Length", avg_length, e)
            writer.add_scalar("Average Loss", avg_loss, e)
            writer.add_scalar("Average Q-Value", avg_q, e)

    # Play
    elif args.mode == "play":
        print("Play mode activated.")
        state, info = env.reset() # Initialize environment
        total_reward = 0
        # Main game loop
        while True:
            actionId = mario.act(state, train_mode=False)  # Select action using the model
            if args.debug == True:
                    ai_interface.PrintGameState(pyboy)
            action = filteredActions[actionId]
            next_state, reward, done, truncated, info = env.step(action)  # Apply action
            total_reward += reward
            state = next_state # Update state
            if done:
                break # Exit loop if game ends
        print(f"Total score: {total_reward}") # Display final score

    # Cleanup resources
    env.close()
    if args.mode == "train":
        writer.close()
    exit() # Terminate script