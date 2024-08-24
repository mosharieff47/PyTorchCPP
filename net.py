import torch
import torch.nn as nn


class NeuralNet(nn.Module):

    def __init__(self, inputs, outputs):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(inputs, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, outputs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

