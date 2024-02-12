import torch
from tqdm import tqdm
import numpy as np
from deep_q import DDQN

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class MarioGymAI():

    def __init__(self, state_dim, action_space_dim, save_dir, args):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_dim = state_dim
        self.action_space_dim = action_space_dim
        self.save_dir = save_dir
        self.args = args # argparse
    
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

        # Q learning
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        if self.args.lr_scheduler == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.exploration_rate_decay)
            print(f"scheduler set to {self.args.lr_scheduler}")
        elif self.args.lr_scheduler == "Cyclic":
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
                                                               base_lr=0.0001, max_lr=0.001, step_size_up=2000, 
                                                               mode='triangular', cycle_momentum=False)
            print(f"scheduler set to {self.args.lr_scheduler}")

        self.loss_fn = torch.nn.SmoothL1Loss()
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
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        if not train_mode:
            # Im Spielmodus: Wähle die Aktion mit dem höchsten Q-Wert
            state = np.array(state, dtype=np.float32)
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            with torch.no_grad():  # Keine Notwendigkeit für Gradientenberechnung
                action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()
        else:
            # Im Trainingsmodus: Logik für Exploration und Exploitation
            # EXPLORE
            if (np.random.rand() < self.exploration_rate):
                action_idx = np.random.randint(0, self.action_space_dim)
            # EXPLOIT
            else:
                state = np.array(state, dtype=np.float32)
                state = torch.tensor(state, device=self.device).unsqueeze(0)
                # model action
                action_values = self.net(state, model="online")
                action_idx = torch.argmax(action_values, axis=1).item()

            # decrease exploration_rate
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)

        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    # Building qlearning network
    def td_estimate(self, state, action):
        model_output = self.net(state, model="online")
        current_Q = model_output[np.arange(0, self.batch_size), action]  # Q_online(s,a)
        return current_Q

    @torch.no_grad() # Disable gradient calculations here (because no need to backpropagate "target")
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        global total_loss, cnt
        """Update online action value (Q) function with a batch of experiences"""
        self.pbar.set_postfix_str("\U0001F4BE ")
        self.pbar.update(1)
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # Save every end of sequence
        if self.curr_step % self.save_every == 0:
            self.save()
            self.pbar.reset()
            self.pbar.total = self.save_every

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD 
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
    
    def update_moving_averages(self, reward, length, loss, q_value):
        self.moving_avg_ep_rewards.append(reward)
        self.moving_avg_ep_lengths.append(length)
        self.moving_avg_ep_avg_losses.append(loss)
        self.moving_avg_ep_avg_qs.append(q_value)

        if len(self.moving_avg_ep_rewards) > 100:  # Calculate moving averages over the last 100 episodes
            self.moving_avg_ep_rewards.pop(0)
            self.moving_avg_ep_lengths.pop(0)
            self.moving_avg_ep_avg_losses.pop(0)
            self.moving_avg_ep_avg_qs.pop(0)

        self.avg_reward = sum(self.moving_avg_ep_rewards) / len(self.moving_avg_ep_rewards)
        self.avg_length = sum(self.moving_avg_ep_lengths) / len(self.moving_avg_ep_lengths)
        self.avg_loss = sum(self.moving_avg_ep_avg_losses) / len(self.moving_avg_ep_avg_losses)
        self.avg_q = sum(self.moving_avg_ep_avg_qs) / len(self.moving_avg_ep_avg_qs)

        return self.avg_reward, self.avg_length, self.avg_loss, self.avg_q
    
    def load_model(self, path):
        dt = torch.load(path, map_location=torch.device(self.device))
        self.net.load_state_dict(dt["model"])
        if self.args.load_optimizer_state == True:
            try:
                self.optimizer.load_state_dict(dt["optimizer"])
                self.scheduler.load_state_dict(dt["scheduler"])
                print("Optimizer and Scheduler states loaded successfully.")
            except KeyError as e:
                print(f"Error loading optimizer or scheduler state: {e} (No such key found)")

        self.exploration_rate = dt["exploration_rate"]
        self.curr_step = dt.get("curr_step", 0) # Default value 0 as fallback
        print(f"Loading step {self.curr_step}, with exploration rate {self.exploration_rate}")
    
    def save(self):

        save_path = f"{self.save_dir}/mario_net_step_{self.curr_step}_exp_{self.exploration_rate:.3f}_avgr_{self.avg_reward:.3f}.chkpt"
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