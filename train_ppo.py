#Note, much of this code was from the CS Deep Learning class taught at BYU that Drew Sumsion took
# Drew Sumsion is writing this and acknowledging the help and gratitude for offering this code for me to build off of.

#Note, this was originally in colab. So, if you want to run each section at a time I have put in this where it requires user input to move to next section. I called it "colabMode"
from datetime import datetime
from pathlib import Path

from simulation import simulation

colabMode = False

#imports
# ! pip3 install gym #maybe might not need this one....
# ! pip3 install torch

if colabMode:
    print("about to start init.")
    input("Waiting for user input to continue....")

import torch
import torch.nn as nn
from itertools import chain
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 0
torch.manual_seed(seed)


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
    state_tensor = torch.Tensor([state]).to(device)
    action_dist = network(state_tensor)

    # now we pick one action from the action distribution
    # action = torch.multinomial(action_dist, 1).item()
    action = torch.argmax(action_dist, 1).item()

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
    policy.reset()
    epoch_cntr = 0
    for epoch in range(0, policy_epochs):
        cntr = 0
        epoch_cntr += 1
        for state, action, action_dist, theReturn in memory_dataloader:
            cntr += 1
            optim.zero_grad()

            # first set ups
            state = state.float().to(device)
            theReturn = theReturn.float().to(device)
            action = action.to(device)
            action_dist = action_dist.to(device)
            action_dist = action_dist.detach()

            # get the value loss
            value_loss = nn.functional.mse_loss(value(state).squeeze(), theReturn)

            # advantage for the policy loss
            advantage = theReturn - value(state)  # actual - expected
            advantage = advantage.detach()

            # policy loss
            encoded = nn.functional.one_hot(action, num_classes=policy(state).shape[1]).bool()
            policy_ratio = policy(state)[encoded] / action_dist.squeeze()[encoded]

            # prevent overfitting
            clip_policy_ratio = torch.clamp(policy_ratio, 1 - epsilon, 1 + epsilon)
            policy_loss = -1 * torch.mean(torch.minimum(policy_ratio * advantage, clip_policy_ratio * advantage))

            # combine the loss
            loss = value_loss + policy_loss

            # normal dep learning things
            loss.backward()
            optim.step()
            policy.reset()

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
        self.hidden_size = 450#8
        self.n_layers = 2
        bidirectional = False
        self.D = 1 #Note, this must be 2 if bidirectional=True or 1 if bidirectional=False

        self.net = nn.GRU(input_size=state_size, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=False, bidirectional=bidirectional)
        self.linear1 = nn.Linear(self.hidden_size, action_size)

        self.reset()

    def reset(self):
        self.previous_last_layer = None

    def forward(self, x):
        """Get policy from state

          Args:
              state (tensor): current state, size (batch x state_size)

          Returns:
              action_dist (tensor): probability distribution over actions (batch x action_size)
        """
        x = torch.unsqueeze(x, 0)
        if self.previous_last_layer is None:
            batchSize = x.shape[1]
            last_layer_shape = (self.D * self.n_layers, batchSize, self.hidden_size)
            self.previous_last_layer = torch.zeros(last_layer_shape)

        output, gru_final_layer = self.net(x, self.previous_last_layer)
        self.previous_last_layer = gru_final_layer
        linear_out = self.linear1(gru_final_layer[-1])
        lin_out_norm = nn.functional.normalize(linear_out, dim=1)#torch.unsqueeze(linear_out, 0))
        return lin_out_norm


# Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        hidden_size = 450#8

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
    lr = 1e-6
    epochs = 100
    saveEvery_epochs = 1
    env_samples = 100
    gamma = 0.9
    batch_size = 256
    epsilon = 0.2
    policy_epochs = 5

    # Init environment
    state_size = 225#4
    action_size = 60#2
    env = simulation()#gym.make('CartPole-v1')

    # Init networks
    policy_network = PolicyNetwork(state_size, action_size).to(device)
    # policy_network.load_state_dict(
    #     torch.load("policy_epoch_99.pt",
    #                map_location=device))
    value_network = ValueNetwork(state_size).to(device)

    # Init optimizer
    optim = torch.optim.Adam(chain(policy_network.parameters(), value_network.parameters()), lr=lr)

    savePath = None

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
            while not done and cum_reward < 3000:  # End after 20000 steps
                policy_network.reset()

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
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        policy_network.reset()
        learn_ppo(optim, policy_network, value_network, loader, epsilon, policy_epochs)

        # Print results
        results_ppo.extend(rewards)  # Store rewards for this epoch
        loop.update(1)
        loop.set_description("Epochs: {} Reward: {}".format(epoch, results_ppo[-1]))

        if not (epoch % saveEvery_epochs):
            if savePath is None:
                savePath = Path("./Results")
                savePath.mkdir(exist_ok=True)
                # create directory to save in
                saveDir = str(datetime.now())
                saveDir = saveDir.replace(" ", "__")
                saveDir = saveDir.replace("-", "_")
                saveDir = saveDir.replace(":", "_")
                saveDir = saveDir.replace(".", "_")
                savePath = savePath / Path(saveDir)
                savePath.mkdir()

            policyPath = savePath / ("policy_epoch_" + str(epoch) + ".pt")
            valuePath = savePath / ("value_epoch_" + str(epoch) + ".pt")
            plotPath = savePath / ("plot_epoch_" + str(epoch) + ".png")

            torch.save(policy_network.state_dict(), str(policyPath))
            torch.save(value_network.state_dict(), str(valuePath))

            plt.plot(results_ppo)
            plt.xlabel("Epoch")
            plt.ylabel("Reward")
            # plt.show()
            plt.savefig(plotPath)

    return results_ppo


results_ppo = ppo_main()

