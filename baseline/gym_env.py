import torch
import numpy as np
import random
from collections import deque
from deep_q import DDQN

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

total_loss = 0
cnt = 0

class MarioGymAI():

    def __init__(self, state_dim, action_space_dim, save_dir):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_dim = state_dim
        self.action_space_dim = action_space_dim
        self.save_dir = save_dir
    
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
        self.save_every = 1e4

        # Q learning
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.exploration_rate_decay)
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
            # Im Trainingsmodus: Bestehende Logik für Exploration und Exploitation
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

    @torch.no_grad() # Disable gradient calculations here (because we don’t need to backpropagate "target")
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
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # Save every end of sequence
        #if self.curr_step % self.save_every == 0:
        #    self.save()

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
        #total_loss += loss
        #cnt += 1
        #print(f"step:{self.curr_step}")
        #if cnt == 200:
            #print(total_loss / cnt)
            #total_loss = 0
            #cnt = 0

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

        avg_reward = sum(self.moving_avg_ep_rewards) / len(self.moving_avg_ep_rewards)
        avg_length = sum(self.moving_avg_ep_lengths) / len(self.moving_avg_ep_lengths)
        avg_loss = sum(self.moving_avg_ep_avg_losses) / len(self.moving_avg_ep_avg_losses)
        avg_q = sum(self.moving_avg_ep_avg_qs) / len(self.moving_avg_ep_avg_qs)

        return avg_reward, avg_length, avg_loss, avg_q
    
    def load_model(self, path):
        dt = torch.load(path, map_location=torch.device(self.device))
        self.net.load_state_dict(dt["model"])
        self.exploration_rate = dt["exploration_rate"]
        print(f"Loading model at {path} with exploration rate {self.exploration_rate}")
    
    def save(self):
        save_path = (f"{self.save_dir}/mario_net_step_{int(self.curr_step)}.chkpt")
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")