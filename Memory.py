import torch


class Buffer(object):
    def __init__(self, device, capacity=1500):
        super(Buffer).__init__()
        print(device)
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.actions = []
        self.is_terminal = []
        self.capacity = capacity
        self.device = device

    def add_transition(self, state, action, reward, log_prob, is_terminal):
        self.states.append(state)
        self.actions.append(action.detach())
        self.rewards.append(reward)
        self.logprobs.append(log_prob.detach())
        self.is_terminal.append(is_terminal)

        return len(self.states) == self.capacity

    def get_transitions(self):
        old_states = torch.tensor(self.states, dtype=torch.float).to(self.device)
        old_actions = torch.cat(self.actions).to(self.device)
        old_prob = torch.cat(self.logprobs).to(self.device)

        return old_states, old_actions, old_prob, self.rewards, self.is_terminal

    def clear(self):
        del self.states[:]
        del self.rewards[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.actions[:]
        del self.is_terminal[:]
