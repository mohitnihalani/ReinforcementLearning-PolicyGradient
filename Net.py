import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, input_dim=4, output_dim=2, hidden_dim=128, dropout=0.3):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(128, output_dim)


    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, actions, hidden_dim=128):
        super(ActorCritic, self).__init__()

        #Actor
        self.actor = Network(4, actions)
        self.critic = Network(4, 1)

    def forward(self, x):
        action_logits = self.actor(x)
        critic_value = self.critic(x).squeeze()
        return action_logits, critic_value
