"""
Adapted from: https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/model.py

The code was modified to add one more hidden layer as suggested by the paper: 
Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising
(https://arxiv.org/pdf/1802.08365.pdf)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=100, 
                    fc2_units=100, fc3_units=100):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
