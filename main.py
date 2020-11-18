# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import gym
import torch
from Reinforce import Reinforce
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PPO import PPO

sns.set_theme()

SEED = 1234

def calculate_running_average(rewards, episode=50):
    N = len(rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(rewards[max(0, t - 50):(t + 1)])
    return running_avg


def plot_evaluation_graph(rewards, algorithm, ylabel):
    print("Plotting")
    fig = plt.figure()
    x = np.arange(len(rewards))
    running_average = calculate_running_average(rewards,20)
    plt.plot(x, rewards, label="Scores (original)")
    plt.plot(x, running_average, color="C1", label="Moving Average (20)")
    plt.xlabel('Episodes')
    plt.ylabel('Episodic {0}: Evaluation'.format(ylabel))
    plt.title("Policy Gradient Using {0}".format(algorithm))
    plt.legend()
    plt.savefig("PolicyGradientEvaluation-{0}-{1}.png".format(algorithm, ylabel))
    plt.show()


def plot_graph(rewards, algorithm, ylabel):
    print("Plotting")

    x = np.arange(len(rewards))
    running_average = calculate_running_average(rewards)
    plt.plot(x, rewards,label="{0} (original)".format(ylabel))
    plt.plot(x, running_average, color="C1", label="Moving Average (20)")
    plt.xlabel('Episodes')
    plt.ylabel('Episodic {0} While Training'.format(ylabel))
    plt.title("Policy Gradient Using {0}".format(algorithm))
    plt.legend()
    plt.savefig("PolicyGradient-{0}-{1}.png".format(algorithm, ylabel))
    plt.show()


if __name__ == '__main__':
    print("Starting")

    agentType = "REINFORCE"
    episodes = 1000
    max_steps = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('CartPole-v1')
    in_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    env.seed(SEED)
    if agentType == "REINFORCE":
        agent = Reinforce(num_actions, device)
    else:
        agent = PPO(num_actions, device)
        episodes = 150

    evaluate_episodes = 100
    rewards = []
    episodic_loss = []

    for episode in range(episodes):
        episodic_rewards, loss = agent.run_episode(env, True)
        print("Episode: {0}  Rewards: {1}".format(episode, episodic_rewards))
        rewards.append(episodic_rewards)
        episodic_loss.append(loss)
        if np.mean(rewards) > 475 :
            break

    plot_graph(rewards,agentType, "Rewards")
    plot_graph(episodic_loss, agentType, "Loss")

    evaluation_rewards = []
    for episode in range(evaluate_episodes):
        episodic_rewards ,_= agent.run_episode(env, False)
        evaluation_rewards.append(episodic_rewards)

    plot_evaluation_graph(evaluation_rewards, agentType, "Rewards")

