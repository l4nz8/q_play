import torch
from tqdm import tqdm
import numpy as np
from deep_q import DDQN

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class MarioGymAI():

    def __init__(self, state_dim, action_space_dim, save_dir, args):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # Determine GPU or CPU for training
        self.state_dim = state_dim # Dimensions of the input state to the network
        self.action_space_dim = action_space_dim # Number of possible actions
        self.save_dir = save_dir # Checkpoints Directory
        self.args = args # Command-line arguments
    
        # DDQN algorithm (uses two ConvNets - online and target that independently approximate the optimal action-value function)
        self.net = DDQN(self.state_dim, self.action_space_dim).float()
        self.net = self.net.to(device=self.device)

        # network parameter
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # Memory
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32
        self.save_every = int(1e5)
        self.pbar = tqdm(total=self.save_every,desc="Saving Progress", colour="#FF0000")

        # Q learning hyperparameters
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025) # Optimizer
        # Learning rate scheduler setup
        if self.args.lr_scheduler == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.exploration_rate_decay)
            print(f"scheduler set to {self.args.lr_scheduler}")
        elif self.args.lr_scheduler == "Cyclic":
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
                                                               base_lr=0.0001, max_lr=0.001, step_size_up=2000, 
                                                               mode='triangular', cycle_momentum=False)
            print(f"scheduler set to {self.args.lr_scheduler}")

        self.loss_fn = torch.nn.SmoothL1Loss() # Loss function
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

        # Metric lists
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

    def act(self, state, train_mode=True):
        """
        Given a state, choose an epsilon-greedy action and update value of step

        Args:
        - state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        - train_mode(bool): Flag to switch between training and playing mode.

        Returns:
        - action_idx(int): Index number of the selected action
        """
        if not train_mode:
            # Play mode: Select action with highest predicted Q-value (exploitation)
            state = np.array(state, dtype=np.float32)
            state = torch.tensor(state, device=self.device).unsqueeze(0) # Convert state to tensor for model input
            with torch.no_grad():  # Disable gradient computation for inference
                action_values = self.net(state, model="online") # Model get action
            action_idx = torch.argmax(action_values, axis=1).item()
        else:
            # Training mode: Use epsilon-greedy policy for exploration-exploitation trade-off
            # EXPLORE - Randomly select any action
            if (np.random.rand() < self.exploration_rate):
                action_idx = np.random.randint(0, self.action_space_dim)
            # EXPLOIT - Select action with highest predicted Q-value
            else:
                state = np.array(state, dtype=np.float32)
                state = torch.tensor(state, device=self.device).unsqueeze(0) # Convert state to tensor for model input
                action_values = self.net(state, model="online") # Model get action
                action_idx = torch.argmax(action_values, axis=1).item()

            # Decay exploration rate to gradually reduce random actions
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step, used for tracking within episode
        self.curr_step += 1
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Args:
        - state (LazyFrame), environment's previous state
        - next_state (LazyFrame), environment's next state after taking the action
        - action (int), action taken in the state
        - reward (float), reward received after taking the action
        - done(bool), indicating whether the episode has ende
        """
        # Convert inputs to PyTorch tensors and store them in the replay buffer
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)

        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        # Add the experience to the memory
        self.memory.add(TensorDict({
            "state": state, 
            "next_state": next_state, 
            "action": action, 
            "reward": reward, 
            "done": done
            }, batch_size=[]))

    def recall(self):
        """
        Retrieve a batch of experiences from memory (replay buffer)

        Returns:
        - A batch of experiences including states, actions, rewards, and done signals
        """
        # Sample a batch from memory and move it to the current device
        batch = self.memory.sample(self.batch_size).to(self.device)
        # Extract tensors for each component of the experience
        state, next_state, action, reward, done = (batch.get(key) for key in("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def td_estimate(self, state, action):
        """
        Calculates the Temporal Difference (TD) estimate

        Args:
        - state: The state tensor
        - action: The action tensor

        Returns:
        - The estimated Q values for the given state-action pairs
        """
        # Get Q values for all actions in given states
        model_output = self.net(state, model="online")
        # Select the Q value for the taken action
        current_Q = model_output[np.arange(0, self.batch_size), action]  # Q_online(sate,action)
        return current_Q

    @torch.no_grad() # Disable gradient calculations (because no need to backpropagate "target")
    def td_target(self, reward, next_state, done):
        """
        Calculates the TD target for updating the Q value

        Args:
        - reward: The reward tensor
        - next_state: The next state tensor
        - done: The done tensor indicating episode termination

        Returns:
        - The target Q value for the given state-action pairs
        """
        # Predict next Q values using the online model to select actions
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        # Calculate target Q value with the target model
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        # Apply the Bellman equation
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        """
        Updates the online Q network based on the TD estimate and TD target

        Args:
        - td_estimate: The estimated Q values
        - td_target: The target Q values

        Returns:
        - The loss value as a result of the update
        """
        # Calculate loss
        loss = self.loss_fn(td_estimate, td_target)
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Update online network parameters
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()

    def sync_Q_target(self):
        """
        Synchronizes the target Q network with the online Q network
        """
        # Copy parameters from the online network to the target network
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        """Updates the online Q-network using a batch of experiences from the replay buffer"""
        # Update the progress bar (if applicable)
        self.pbar.set_postfix_str("\U0001F4BE ")
        self.pbar.update(1)

        # Sync the target Q-network with the online Q-network at specified intervals
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # Save the model at specified intervals
        if self.curr_step % self.save_every == 0:
            self.save()
            self.pbar.reset()
            self.pbar.total = self.save_every

        # Check if enough steps have been taken to start learning
        if self.curr_step < self.burnin:
            return None, None # Not enough data to learn yet

        # Only learn at specified intervals
        if self.curr_step % self.learn_every != 0:
            return None, None  # Not time to learn yet

        # Sample a batch of experiences from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD estimate
        td_est = self.td_estimate(state, action)
        # Get TD target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate the online Q-network and return the average Q-value and the loss
        loss = self.update_Q_online(td_est, td_tgt)
        return (td_est.mean().item(), loss)
    
    def update_moving_averages(self, reward, length, loss, q_value):
        """Updates moving averages of rewards, lengths, losses, and Q-values."""
        # Append recent episode metrics to their respective lists
        self.moving_avg_ep_rewards.append(reward)
        self.moving_avg_ep_lengths.append(length)
        self.moving_avg_ep_avg_losses.append(loss)
        self.moving_avg_ep_avg_qs.append(q_value)

        # Keep only the last 100 episodes for moving averages
        if len(self.moving_avg_ep_rewards) > 100:
            self.moving_avg_ep_rewards.pop(0)
            self.moving_avg_ep_lengths.pop(0)
            self.moving_avg_ep_avg_losses.pop(0)
            self.moving_avg_ep_avg_qs.pop(0)

        # Calculate the moving averages
        self.avg_reward = sum(self.moving_avg_ep_rewards) / len(self.moving_avg_ep_rewards)
        self.avg_length = sum(self.moving_avg_ep_lengths) / len(self.moving_avg_ep_lengths)
        self.avg_loss = sum(self.moving_avg_ep_avg_losses) / len(self.moving_avg_ep_avg_losses)
        self.avg_q = sum(self.moving_avg_ep_avg_qs) / len(self.moving_avg_ep_avg_qs)

        return self.avg_reward, self.avg_length, self.avg_loss, self.avg_q
    
    def load_model(self, path):
        """Load model with its optimizer, scheduler, and other parameters."""
        # Load the checkpoint
        dt = torch.load(path, map_location=torch.device(self.device))
        self.net.load_state_dict(dt["model"])

        # Attempt to load optimizer and scheduler states if applicable
        if self.args.load_optimizer_state == True:
            try:
                self.optimizer.load_state_dict(dt["optimizer"])
                self.scheduler.load_state_dict(dt["scheduler"])
                print("Optimizer and Scheduler states loaded successfully.")
            except KeyError as e:
                print(f"Error loading optimizer or scheduler state: {e} (No such key found)")

        # Load other training parameters
        self.exploration_rate = dt["exploration_rate"]
        self.curr_step = dt.get("curr_step", 0) # Default value 0 as fallback
        print(f"Loading step {self.curr_step}, with exploration rate {self.exploration_rate}")
    
    def save(self):
        """Saves the current model along with its optimizer, scheduler, and other parameters."""
        # Construct the save path
        save_path = f"{self.save_dir}/mario_net_step_{self.curr_step}_exp_{self.exploration_rate:.3f}_avgr_{self.avg_reward:.3f}.chkpt"
        # Save the checkpoint
        torch.save(
            dict(model=self.net.state_dict(), 
                 optimizer=self.optimizer.state_dict(),
                 scheduler=self.scheduler.state_dict(),
                 exploration_rate=self.exploration_rate,
                 curr_step=self.curr_step,
                 avg_reward=self.avg_reward,
                 avg_loss=self.avg_loss
                 ),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")