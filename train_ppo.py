#Note, much of this code was from the CS Deep Learning class taught at BYU that Drew Sumsion took
# Drew Sumsion is writing this and acknowledging the help and gratitude for offering this code for me to build off of.

#Note, this was originally in colab. So, if you want to run each section at a time I have put in this where it requires user input to move to next section. I called it "colabMode"
colabMode = True

#imports
# ! pip3 install gym #maybe might not need this one....
# ! pip3 install torch

if colabMode:
    print("about to start init.")
    input("Waiting for user input to continue....")

import gym
import torch
import torch.nn as nn
from itertools import chain
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np

if colabMode:
    print("About to start Part 2: PPO")
    input("Waiting for user input to continue....")

def calculate_return(memory, rollout, gamma):
    """Return memory with calculated return in experience tuple

      Args:
          memory (list): (state, action, action_dist, return) tuples
          rollout (list): (state, action, action_dist, reward) tuples from last rollout
          gamma (float): discount factor

      Returns:
          list: memory updated with (state, action, action_dist, return) tuples from rollout
    """
    firstTime = True
    previous = None
    newMem = []
    for newState, newAction, newAction_dist, reward in reversed(rollout):

        if firstTime:
            new_return = reward
            firstTime = False
        else:
            new_return = reward + previous * gamma
        previous = new_return
        newMem.append([newState, newAction, newAction_dist, new_return])

    for x in reversed(newMem):
        memory.append(x)

    return memory


def get_action_ppo(network, state):
    """Sample action from the distribution obtained from the policy network

      Args:
          network (PolicyNetwork): Policy Network
          state (np-array): current state, size (state_size)

      Returns:
          int: action sampled from output distribution of policy network
          array: output distribution of policy network
    """
    # run the state through the network
    state_tensor = torch.Tensor([state]).cuda()
    action_dist = network(state_tensor)

    # now we pick one action from the action distribution
    action = torch.multinomial(action_dist, 1).item()

    return action, action_dist


def learn_ppo(optim, policy, value, memory_dataloader, epsilon, policy_epochs):
    """Implement PPO policy and value network updates. Iterate over your entire
       memory the number of times indicated by policy_epochs.

      Args:
          optim (Adam): value and policy optimizer
          policy (PolicyNetwork): Policy Network
          value (ValueNetwork): Value Network
          memory_dataloader (DataLoader): dataloader with (state, action, action_dist, return, discounted_sum_rew) tensors
          epsilon (float): trust region
          policy_epochs (int): number of times to iterate over all memory
    """
    for epoch in range(0, policy_epochs):
        for state, action, action_dist, theReturn in memory_dataloader:
            optim.zero_grad()

            # first set ups
            state = state.float().cuda()
            theReturn = theReturn.float().cuda()
            action = action.cuda()
            action_dist = action_dist.cuda()
            action_dist = action_dist.detach()

            # get the value loss
            value_loss = nn.functional.mse_loss(value(state).squeeze(), theReturn)

            # advantage for the policy loss
            advantage = theReturn - value(state)  # actual - expected
            advantage = advantage.detach()

            # policy loss
            encoded = nn.functional.one_hot(action).bool()
            policy_ratio = policy(state)[encoded] / action_dist.squeeze()[encoded]

            # prevent overfitting
            clip_policy_ratio = torch.clamp(policy_ratio, 1 - epsilon, 1 + epsilon)
            policy_loss = -1 * torch.mean(torch.minimum(policy_ratio * advantage, clip_policy_ratio * advantage))

            # combine the loss
            loss = value_loss + policy_loss

            # normal dep learning things
            loss.backward()
            optim.step()

if colabMode:
    print("About to start Modules")
    input("Waiting for user input to continue....")

# Dataset that wraps memory for a dataloader
class RLDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = []
        for d in data:
            self.data.append(d)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        hidden_size = 8

        self.net = nn.Sequential(nn.Linear(state_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, action_size),
                                 nn.Softmax(dim=1))

    def forward(self, x):
        """Get policy from state

          Args:
              state (tensor): current state, size (batch x state_size)

          Returns:
              action_dist (tensor): probability distribution over actions (batch x action_size)
        """
        return self.net(x)


# Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        hidden_size = 8

        self.net = nn.Sequential(nn.Linear(state_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        """Estimate value given state

          Args:
              state (tensor): current state, size (batch x state_size)

          Returns:
              value (tensor): estimated value, size (batch)
        """
        return self.net(x)

if colabMode:
    print("About to start Main")
    input("Waiting for user input to continue....")

def ppo_main():
    # Hyper parameters
    lr = 1e-3
    epochs = 20
    env_samples = 100
    gamma = 0.9
    batch_size = 256
    epsilon = 0.2
    policy_epochs = 5

    # Init environment
    state_size = 4
    action_size = 2
    env = gym.make('CartPole-v1')

    # Init networks
    policy_network = PolicyNetwork(state_size, action_size).cuda()
    value_network = ValueNetwork(state_size).cuda()

    # Init optimizer
    optim = torch.optim.Adam(chain(policy_network.parameters(), value_network.parameters()), lr=lr)

    # Start main loop
    results_ppo = []
    loop = tqdm(total=epochs, position=0, leave=False)
    for epoch in range(epochs):

        memory = []  # Reset memory every epoch
        rewards = []  # Calculate average episodic reward per epoch

        # Begin experience loop
        for episode in range(env_samples):

            # Reset environment
            state = env.reset()
            done = False
            rollout = []
            cum_reward = 0  # Track cumulative reward

            # Begin episode
            while not done and cum_reward < 200:  # End after 200 steps
                # Get action
                action, action_dist = get_action_ppo(policy_network, state)

                # Take step
                next_state, reward, done, _ = env.step(action)
                # env.render()

                # Store step
                rollout.append((state, action, action_dist, reward))

                cum_reward += reward
                state = next_state  # Set current state

            # Calculate returns and add episode to memory
            memory = calculate_return(memory, rollout, gamma)

            rewards.append(cum_reward)

        # Train
        dataset = RLDataset(memory)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        learn_ppo(optim, policy_network, value_network, loader, epsilon, policy_epochs)

        # Print results
        results_ppo.extend(rewards)  # Store rewards for this epoch
        loop.update(1)
        loop.set_description("Epochs: {} Reward: {}".format(epoch, results_ppo[-1]))

    return results_ppo


results_ppo = ppo_main()

plt.plot(results_ppo)
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.show()