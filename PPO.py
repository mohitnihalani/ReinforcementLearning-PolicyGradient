import torch
import torch.nn as nn
from Net import ActorCritic
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from Memory import Buffer
import numpy as np


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

class PPO(object):
    def __init__(self, actions, device, lr=1e-2, gamma=0.99, ppo_clip=0.2, ppo_epoch=5, batch_size=8):
        super(PPO).__init__()
        self.device = device
        self.policy_new = ActorCritic(actions)
        self.policy_old = ActorCritic(actions)
        self.policy_new.apply(init_weights)
        self.policy_old.apply(init_weights)
        self.optimizer = optim.Adam(self.policy_new.parameters(), lr=lr)
        self.policy_old.load_state_dict(self.policy_new.state_dict())
        self.ppo_clip = ppo_clip
        self.ppo_epoch = ppo_epoch
        self.batch_size = batch_size
        self.memory = Buffer(device)
        self.max_timestamp_per_episode = 1000
        self.gamma = gamma
        self.value_loss_coef = 0.5
        self.MseLoss = nn.MSELoss()

    def get_action_and_prob(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_predictions, _ = self.policy_old(state)
        action_probabilities = F.softmax(action_predictions, dim=-1)
        distributions = Categorical(action_probabilities)
        action = distributions.sample()
        action_log_prob = distributions.log_prob(action)
        return action, action_log_prob

    def evaluate_policy(self, states, actions):
        action_predictions, value_pred = self.policy_new(states)
        action_probabilities = F.softmax(action_predictions, dim=-1)
        distributions = Categorical(action_probabilities)
        action_log_probs = distributions.log_prob(actions)
        return action_log_probs, value_pred

    def calculate_rewards(self, old_rewards):

        rewards = []
        discounted_reward = 0
        for reward in reversed(old_rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.append(discounted_reward)
        rewards.reverse()
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        return rewards

    def calculate_advantage(self, discounted_rewards, critic_rewards):
        advantages = discounted_rewards - critic_rewards
        advantages = (advantages - advantages.mean()) / advantages.std()
        return advantages

    def run_episode(self, env, train):
        state = env.reset()
        total_rewards = 0
        loss = 0
        for i in range(self.max_timestamp_per_episode):
            if not train:
                env.render()
            action, action_log_prob = self.get_action_and_prob(state)
            next_state, reward, done, _ = env.step(action.item())
            total_rewards += reward
            if train:
                self.memory.add_transition(state, action, reward, action_log_prob, done)
            state = next_state
            if done:
                break
        if train:
            loss = self.train_policy()
            self.memory.clear()
        return total_rewards, loss

    def train_policy(self):

        old_states, old_actions, old_probs, rewards, _ = self.memory.get_transitions()
        rewards = self.calculate_rewards(rewards)
        epoch_loss = []
        for epoch in range(self.ppo_epoch):
            probs, value_pred = self.evaluate_policy(old_states, old_actions)
            ratio = (probs - old_probs).exp()
            advantages = self.calculate_advantage(rewards, value_pred.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, min = 1.0 - self.ppo_clip, max = 1.0 + self.ppo_clip) * advantages
            self.optimizer.zero_grad()
            loss = (-torch.min(surr1, surr2) + self.value_loss_coef * F.smooth_l1_loss(value_pred, rewards)).mean()
            loss.backward()
            self.optimizer.step()
            epoch_loss.append(loss.item())

        self.policy_old.load_state_dict(self.policy_new.state_dict())

        return np.mean(epoch_loss)
