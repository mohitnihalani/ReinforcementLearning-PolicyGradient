import torch
from Net import Network
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class Reinforce(object):
    def __init__(self, actions, device, lr=1e-2, gamma=0.99):
        super(Reinforce, self).__init__()
        self.device = device
        self.policy = Network(4,actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []
        self.gamma = gamma
        self.max_steps = 10000

    def get_action_and_prob(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_predictions = self.policy(state).cpu()
        action_probabilities = F.softmax(action_predictions, dim=-1)
        distributions = Categorical(action_probabilities)
        action = distributions.sample()
        return action.item(), distributions.log_prob(action)

    def run_episode(self, env, train):
        state = env.reset()
        total_rewards = 0
        loss = 0
        for times_step in range(self.max_steps):
            if not train:
                env.render()
            action, log_prob = self.get_action_and_prob(state)
            state, reward, done, _ = env.step(action)
            total_rewards += reward
            if train:
                self.add_rewards((reward, log_prob))
            if done:
                break
        if train:
            loss = self.train_policy()
        return total_rewards, loss

    def add_rewards(self, data):
        self.memory.append(data)

    def train_policy(self):

        self.policy.train()
        discounted_reward = 0
        self.optimizer.zero_grad()
        policy_loss = []
        for reward, prob in reversed(self.memory):
            discounted_reward = reward + self.gamma * discounted_reward
            policy_loss.append(-prob * discounted_reward)

        torch.tensor(policy_loss)
        policy_loss = torch.cat(policy_loss)
        policy_loss.sum().backward()
        self.optimizer.step()

        del self.memory[:]

        return policy_loss.mean().item()
