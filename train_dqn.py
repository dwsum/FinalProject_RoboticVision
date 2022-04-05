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
    print("\n")
    print("About to start Part 1: DQN")
    input("Waiting for user input to continue....")

def get_action_dqn(network, state, epsilon, epsilon_decay):
    """Select action according to e-greedy policy and decay epsilon

      Args:
          network (QNetwork): Q-Network
          state (np-array): current state, size (state_size)
          epsilon (float): probability of choosing a random action
          epsilon_decay (float): amount by which to decay epsilon

      Returns:
          action (int): chosen action [0, action_size)
          epsilon (float): decayed epsilon
    """
    # if probability of epsilon, do first action, if 1 - epsilon, do second action.
    if np.random.rand(1) < epsilon:
        # exploaration
        action = np.random.randint(2)
    else:
        # explotation
        state = torch.Tensor(state).cuda()
        action = torch.argmax(network(state)).item()

    return action, epsilon * epsilon_decay


def prepare_batch(memory, batch_size):
    """Randomly sample batch from memory
       Prepare cuda tensors

      Args:
          memory (list): state, action, next_state, reward, done tuples
          batch_size (int): amount of memory to sample into a batch

      Returns:
          state (tensor): float cuda tensor of size (batch_size x state_size()
          action (tensor): long tensor of size (batch_size)
          next_state (tensor): float cuda tensor of size (batch_size x state_size)
          reward (tensor): float cuda tensor of size (batch_size)
          done (tensor): float cuda tensor of size (batch_size)
    """
    state = []
    action = []
    next_state = []
    reward = []
    done = []
    selection = random.sample(range(0, len(memory)), batch_size)
    for x in selection:
        state.append(memory[x][0])
        action.append(memory[x][1])
        next_state.append(memory[x][2])
        reward.append(memory[x][3])
        done.append(memory[x][4])
    state = torch.tensor(state).float().cuda()
    action = torch.Tensor(action).float().cuda()
    next_state = torch.tensor(next_state).float().cuda()
    reward = torch.tensor(reward).float().cuda()
    done = torch.tensor(done).float().cuda()

    return (state, action, next_state, reward, done)


def learn_dqn(batch, optim, q_network, target_network, gamma, global_step, target_update):
    """Update Q-Network according to DQN Loss function
       Update Target Network every target_update global steps

      Args:
          batch (tuple): tuple of state, action, next_state, reward, and done tensors
          optim (Adam): Q-Network optimizer
          q_network (QNetwork): Q-Network
          target_network (QNetwork): Target Q-Network
          gamma (float): discount factor
          global_step (int): total steps taken in environment
          target_update (int): frequency of target network update
    """
    optim.zero_grad()

    state, action, next_state, reward, done = batch

    # get needed information for the target
    theMaxes, indicies = torch.max(target_network(next_state), dim=1)
    target = reward + gamma * theMaxes * (1 - done)

    # get the q value along the action dimension.
    action = action.long()
    encoded = nn.functional.one_hot(action).bool()
    actual = q_network(state)[encoded]

    # calculate the loss
    loss = nn.functional.mse_loss(actual, target)

    # still need to add all the normal stuff.
    loss.backward()
    optim.step()

    if global_step % target_update == 0:
        target_network.load_state_dict(q_network.state_dict().copy())


if colabMode:
    print("About to start Modules")
    input("Waiting for user input to continue....")

# Q-Value Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        hidden_size = 8

        self.net = nn.Sequential(nn.Linear(state_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, action_size))

    def forward(self, x):
        """Estimate q-values given state

          Args:
              state (tensor): current state, size (batch x state_size)

          Returns:
              q-values (tensor): estimated q-values, size (batch x action_size)
        """
        return self.net(x)

if colabMode:
    print("About to start Main")
    input("Waiting for user input to continue....")

def dqn_main():
    # Hyper parameters
    lr = 1e-3
    epochs = 750  # 500 #he said he did 750 epochs.
    start_training = 1000
    gamma = 0.99
    batch_size = 32
    epsilon = 1
    epsilon_decay = .9999
    target_update = 1000
    learn_frequency = 2

    # Init environment
    state_size = 4
    action_size = 2
    env = gym.make('CartPole-v1', )

    # Init networks
    q_network = QNetwork(state_size, action_size).cuda()
    target_network = QNetwork(state_size, action_size).cuda()
    target_network.load_state_dict(q_network.state_dict())

    # Init optimizer
    optim = torch.optim.Adam(q_network.parameters(), lr=lr)

    # Init replay buffer
    memory = []

    # Begin main loop
    results_dqn = []
    global_step = 0
    loop = tqdm(total=epochs, position=0, leave=False)
    for epoch in range(epochs):

        # Reset environment
        state = env.reset()
        done = False
        cum_reward = 0  # Track cumulative reward per episode

        # Begin episode
        while not done and cum_reward < 200:  # End after 200 steps
            # Select e-greedy action
            action, epsilon = get_action_dqn(q_network, state, epsilon, epsilon_decay)

            # Take step
            # print(type(env.action_space.sample()))
            next_state, reward, done, _ = env.step(action)
            # env.render()

            # Store step in replay buffer
            memory.append((state, action, next_state, reward, done))

            cum_reward += reward
            global_step += 1  # Increment total steps
            state = next_state  # Set current state

            # If time to train
            if global_step > start_training and global_step % learn_frequency == 0:
                # Sample batch
                batch = prepare_batch(memory, batch_size)

                # Train
                learn_dqn(batch, optim, q_network, target_network, gamma, global_step, target_update)

        # Print results at end of episode
        results_dqn.append(cum_reward)
        loop.update(1)
        loop.set_description('Episodes: {} Reward: {}'.format(epoch, cum_reward))

    return results_dqn


results_dqn = dqn_main()

plt.plot(results_dqn)
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.show()