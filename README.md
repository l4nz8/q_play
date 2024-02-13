![GitHub](https://img.shields.io/github/license/l4nz8/q_play)
# q_play
This project implements a Reinforcement Learning (RL) agent that plays "Super Mario Land" using a Double Deep Q-Network (DDQN). It's built on the PyBoy Game Boy emulator, providing a custom environment for the Mario AI to interact with and learn from. The agent is designed to navigate through the levels of "Super Mario Land," making decisions based on the current state of the game to maximize its reward score and progress.

## Overview
The MARIO-PLAYING RL AGENT uses a DDQN model for decision-making and operates within a custom gym environment tailored around "Super Mario Land". The agent's goal is to learn optimal strategies for navigating the game's levels, overcoming obstacles, and maximizing scores through trial and error.

## Features
- **Double Deep Q-Network:** Utilizes a DDQN architecture for stable and efficient learning.
- **Custom Gym Environment:** Integrates with PyBoy to create a tailored environment for Super Mario Land.
- **Flexible Training Modes:** Supports both training and playing modes for the AI agent.
- **Headless Training:** Offers a headless mode for faster training without rendering the game screen.
- **Customizable Hyperparameters:** Allows tweaking of learning rates, exploration rates, and more.

## Installation
🐍 Python 3.10 is recommended. Other versions may work but have not been tested.

You also need to install Tensorboard and have it aktive in the terminal to monitor the logged metrics.
It also is recommended to use cuda for training, which you must install manually for your individual GPU.

Download and install:
1. Clone the repository to your local machine:
```bash
git clone https://github.com/l4nz8/q_play.git
```
2. Install dependencies:
```bash
cd q_play
pip3 install -r requirements.txt
```
3. Copy your legally obtained Super Mario Land ROM into the gb_ROM/ directory. You can find this using google, it should be 66KB in size. Rename it to `SuperMarioLand.gb` if it is not already. The sha1 sum should be `418203621b887caa090215d97e3f509b79affd3e`, which you can verify by running `shasum SuperMarioLand.gb` inside the terminal. 

Note: the SuperMarioLand.gb file MUST be in the `gb_ROM/` directory and your current directory MUST be the `q_play/` directory (main) in order for this to work.

## Run Emulator and AI

🎮 Run it in the terminal to get help
```bash
python baseline/main.py -h
```

# Based on
DDQN - https://arxiv.org/abs/1509.06461

PyTorch RL Super Mario Bros Example - https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
