import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, device='cuda'):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_shape)
        self.to(self.device)
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, device='cuda'):
        super(CriticNetwork, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.to(self.device)
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
