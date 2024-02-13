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

Note: the SuperMarioLand.gb file MUST be in the `gb_ROM/` directory and your current directory MUST be the `q_play/` root directory in order for this to work.

## Run Emulator and AI:

### 🎮 Run AI 

The script must be started from the root directory `q_play/` in the terminal.
```bash
python baseline/main.py -h
```
The project includes argparse to select start conditions, which are described in more detail with the help of `-h` after initialization of the script.
```usage: main.py [-h] [--world WORLD | --level LEVEL] [-m {train,play}] [--headless] [-ls] [-los] [-lrs {StepLR,Cyclic}] [-exp EXPLORATION] [--debug]```

If the script is started without start conditions, the training mode `-m train` is automatically executed with default settings.

### 📈 Tracking Training Progress: 

To monitor the logged metrics you neet to run Tensorboard from to root directory `q_play/` and have it aktive in the terminal.
```bash
tensorboard --logdir=runs
```
To view tensorboard you have to open `http://localhost:6006/` in your browser (safari is not supported)

Note: updating the progress must be done manually in tensorboard

### 🏋️ Pre-Traind model:

To use a pre-trained model, download the `checkpoints/` folder from the google.docs link below and paste it into the `q_play/` root directory.

**[checkpoints/](https://drive.google.com/drive/folders/1_vqTBNQzlyZl7kOxnsa1q9clRLtB_jYo?usp=sharing)**

the same with the logged metrics

**[runs/](https://drive.google.com/drive/folders/14unJWiTpiiosiZAdMJgtgAOWTQpkiza2?usp=sharing)**

## Based on
### [DDQN](https://arxiv.org/abs/1509.06461)

### [PyTorch RL Super Mario Bros Example](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)

<a href="https://github.com/Baekalfen/PyBoy">
  <img src="/assets/pyboy.svg" height="64">
</a>
