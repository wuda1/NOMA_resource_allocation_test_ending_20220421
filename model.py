# -*- coding:utf-8 -*-
"""
作者：Admin
日期：2022年04月15日
"""
import torch
import torch.nn as nn
import torch.nn.functional as f

N_VEL = 2
N_SUB = 2
state_dimension = N_VEL * N_SUB
num_actions = 2 ** (N_VEL * N_SUB)


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 500)
        nn.init.normal_(self.fc1.weight, 0., 0.1)
        nn.init.constant_(self.fc1.bias, 0.1)
        self.fc2 = nn.Linear(500, 300)
        nn.init.normal_(self.fc2.weight, 0., 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        self.fc3 = nn.Linear(300, 250)
        nn.init.normal_(self.fc3.weight, 0., 0.1)
        nn.init.constant_(self.fc3.bias, 0.1)
        self.fc4 = nn.Linear(250, n_actions)
        nn.init.normal_(self.fc4.weight, 0., 0.1)
        nn.init.constant_(self.fc4.bias, 0.1)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        output = self.fc4(x)
        return output


class ActorNet(nn.Module):
    def __init__(self, state_dim, a_dim):
        super(ActorNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, 100)
        nn.init.normal_(self.fc1.weight, 0., 0.1)
        nn.init.constant_(self.fc1.bias, 0.1)
        self.fc2 = nn.Linear(100, 50)
        nn.init.normal_(self.fc2.weight, 0., 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        self.fc3 = nn.Linear(50, 30)
        nn.init.normal_(self.fc3.weight, 0., 0.1)
        nn.init.constant_(self.fc3.bias, 0.1)
        self.fc4 = nn.Linear(30, a_dim)
        nn.init.normal_(self.fc4.weight, 0., 0.1)
        nn.init.constant_(self.fc4.bias, 0.1)

    def forward(self, x):
        x = f.relu((self.fc1(x)))
        x = f.relu((self.fc2(x)))
        x = f.relu((self.fc3(x)))
        x = f.sigmoid(self.fc4(x))
        return x


class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()

        self.fc1 = nn.Linear(s_dim+a_dim, 100)
        nn.init.normal_(self.fc1.weight, 0., 0.1)
        nn.init.constant_(self.fc1.bias, 0.1)
        self.fc2 = nn.Linear(100, 50)
        nn.init.normal_(self.fc2.weight, 0., 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        self.fc3 = nn.Linear(50, 30)
        nn.init.normal_(self.fc3.weight, 0., 0.1)
        nn.init.constant_(self.fc3.bias, 0.1)
        self.fc4 = nn.Linear(30, 1)
        nn.init.normal_(self.fc4.weight, 0., 0.1)
        nn.init.constant_(self.fc4.bias, 0.1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = f.relu((self.fc1(x)))
        x = f.relu((self.fc2(x)))
        x = f.relu((self.fc3(x)))
        x = self.fc4(x)
        return x
