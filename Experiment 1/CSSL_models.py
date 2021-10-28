import torch.nn.functional as F
import torch.nn as nn


class NN_model(nn.Module):
    def __init__(self):
        super(NN_model, self).__init__()

        self.layer1 = nn.Linear(50, 10)

        self.M_out = nn.Linear(10, 2)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.M_out(x)
        return x


class NN_model_1(nn.Module):
    def __init__(self):
        super(NN_model_1, self).__init__()

        self.layer1 = nn.Linear(50, 40)
        self.layer2 = nn.Linear(40, 40)

        self.M_out1 = nn.Linear(40, 40)
        self.M_out2 = nn.Linear(40, 2)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        x = F.relu(self.M_out1(x))
        x = self.M_out2(x)
        return x


class Representation(nn.Module):
    def __init__(self):
        super(Representation, self).__init__()
        self.layer1 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return x


class output(nn.Module):
    def __init__(self):
        super(output, self).__init__()
        self.M_out = nn.Linear(10, 2)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.M_out(x)
        return x
