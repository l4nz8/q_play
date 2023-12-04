import torch
import numpy as np
import random
from collections import deque
from deep_q import DDQN

total_loss = 0
cnt = 0

class MarioGymAI():

    def __init__(self, state_dim, action_space_dim):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_dim = state_dim
        self.action_space_dim = action_space_dim
    
        # DDQN algorithm (uses two ConvNets - online and target that independently approximate the optimal action-value function)
        self.net = DDQN(self.state_dim, self.action_space_dim)
        self.net = self.net.to(device=self.device)

        # network parameter
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # Memory
        self.memory = deque(maxlen=400000) #100000
        self.batch_size = 4
        self.save_every = 5e5

        # Q learning
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.exploration_rate_decay)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = 10#1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
        
    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if (random.random() < self.exploration_rate):
            action_idx = random.randint(0, self.action_space_dim-1)
        # EXPLOIT
        else:
            state = np.array(state, dtype=np.float64)
            state = torch.tensor(state).to(device=self.device).permute(2,0,1).unsqueeze(0)

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
        state = np.array(state, dtype=np.float64)
        next_state = np.array(next_state, dtype=np.float64)

        state = torch.tensor(state).permute(2,0,1).to(device=self.device)
        next_state = torch.tensor(next_state).permute(2,0,1).to(device=self.device)
        action = torch.tensor([action]).to(device=self.device)
        reward = torch.tensor([reward]).to(device=self.device)
        done = torch.tensor([done]).to(device=self.device)

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def td_estimate(self, state, action):
        model_output = self.net(state, model="online")
        current_Q = model_output[np.arange(0, self.batch_size), action]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
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
        total_loss += loss
        cnt += 1
        if cnt == 200:
            print(total_loss / cnt)
            total_loss = 0
            cnt = 0

        return (td_est.mean().item(), loss)