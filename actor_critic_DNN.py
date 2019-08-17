import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """
    The vanilla nn for target and local network for the actor like in the DDPG Paper
    """
    def __init__(self,state_size, action_size, seed, hid_layer1_size = 128, hid_layer2_size = 128):
        """
        Parameters
        ==========
            state_size (int): Dimension of each state (input features)
            hid_layer1_size (int): Nodes in the first hidden layer
            hid_layer2_size (int): Nodes in the second hidden layer
            action_size (int): Dimension of actions (output)
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size,hid_layer1_size)
        self.fc2 = nn.Linear(hid_layer1_size,hid_layer2_size)
        self.fc3 = nn.Linear(hid_layer2_size,action_size)

        self.bn0 = nn.BatchNorm1d(state_size)
        self.bn1 = nn.BatchNorm1d(hid_layer1_size)
        self.bn2 = nn.BatchNorm1d(hid_layer2_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)


    def forward(self, state):
        """
        Build the feed forward network
        """
        x = self.bn0(state)
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))

        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """
    The vanilla nn for target and local network for the critic like in the DDPG Paper
    """
    def __init__(self,state_size, action_size, seed, hid_layer1_size = 128, hid_layer2_size = 128, hid_layer3_size = 32, hid_layer4_size = 200):
        """
        Parameters
        ==========
            state_size (int): Dimension of each state (input features)
            hid_layer1_size (int): Nodes in the first hidden layer
            hid_layer2_size (int): Nodes in the second hidden layer
            hid_layer3_size (int): Nodes in the third hidden layer
            hid_layer4_size (int): Nodes in the forth hidden layer
            action_size (int): Dimension of actions (output)
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size,hid_layer1_size)
        self.fc2 = nn.Linear(hid_layer1_size + action_size,hid_layer2_size)
        #self.fc3 = nn.Linear(hid_layer2_size,hid_layer3_size)
        #self.fc4 = nn.Linear(hid_layer3_size,hid_layer4_size)
        self.fc5 = nn.Linear(hid_layer2_size, 1)
        
        self.bn0 = nn.BatchNorm1d(state_size)
        self.bn1 = nn.BatchNorm1d(hid_layer1_size)
        self.bn2 = nn.BatchNorm1d(hid_layer2_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        #self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3,3e-3)

    def forward(self, state, action):
        """
        Build the feed forward network
        """
        state = self.bn0(state)
        xs = F.relu(self.fcs1(state))
        x =  torch.cat((xs, action), dim=1) 
        x = F.relu(self.fc2(x))
        #x = F.leaky_relu(self.fc3(x))
        #x = F.relu(self.fc4(x))

        return self.fc5(x)

if __name__ == "__main__":
    ac_test = Actor(3,3,3)
    cr_test = Critic(4,4,4)