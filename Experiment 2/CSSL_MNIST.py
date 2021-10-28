from torch import optim
import torch
import torch.nn.functional as F
import torch.nn as nn
import create_MNIST
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import argparse

torch.manual_seed(123)
parser = argparse.ArgumentParser(description="paraameters for synthetic data, and CSSL")
parser.add_argument("--nL", default=2000, type=int, help="number of labelled data points")
parser.add_argument("--nUL", default=4000, type=int, help="number of unlabelled data points")
parser.add_argument("--bs_L", default=64, type=int, help="batch size for labelled data")
parser.add_argument("--p", default=0.9, type=float, help="p")
parser.add_argument("--lr_L", default=0.1, type=float, help="learning rate for labelled data")
parser.add_argument("--Beta", default=0.01, type=float, help="hyperparamter for regularization term")
args = parser.parse_args()

nL = args.nL
nUL = args.nUL
p = args.p
bs_L = args.bs_L
lr_L = args.lr_L
epochs_L = 30
beta = args.Beta

bs_UL = 64
lr_UL = 1e-3
bs_UL = bs_L * (int(nUL / nL))
if bs_UL == 0:
    bs_UL = 1

x_train_label, y_train_label, x_train_nolabel, y_train_nolabel, x_test, y_test = create_MNIST.MNIST_data(p)
x_train_label = x_train_label[1:nL, :]
y_train_label = y_train_label[1:nL]
x_train_nolabel = x_train_nolabel[1:nUL, :]
y_train_nolabel = y_train_nolabel[1:nUL]

"""
print(y_train_label[1:40])
print("\n\n")
print(y_train_nolabel[1:40])
print("\n\n")
print(y_test[1:50])
"""
train_ds_label = TensorDataset(x_train_label, y_train_label)
train_dl_label = DataLoader(train_ds_label, batch_size=bs_L, shuffle=True)

train_ds = TensorDataset(torch.cat((x_train_label, x_train_nolabel), 0), torch.cat((y_train_label, y_train_nolabel), 0))
train_dl = DataLoader(train_ds, batch_size=bs_UL, shuffle=True)


class Representation(nn.Module):
    def __init__(self):
        super(Representation, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


class output(nn.Module):
    def __init__(self):
        super(output, self).__init__()

        # self.layer1 = nn.Linear(50, 10)
        # self.layer2 = nn.Linear(40, 10)

        self.M_out = nn.Linear(10, 10)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.M_out(x)
        return x


class Mnist_CNN_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
        self.M_out = nn.Linear(10, 10)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        xb = xb.view(-1, xb.size(1))
        return self.M_out(xb)


def loss_func_L(input_pred, target):
    CE = nn.CrossEntropyLoss()
    return CE(input_pred, target)


def loss_func_UL(inputUL, eps=1e-10):
    return (-inputUL * torch.log(inputUL + eps)).sum(1).mean()
    # return torch.sum(-1 * inputLoss * torch.log(inputLoss+eps))


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def fit_L(train_dataL, model, opt, loss_func, epochs):
    for epoch in range(epochs):
        for xb, yb in train_dataL:
            pred = model(xb)
            pred = pred.add(1e-10)
            loss = loss_func(pred, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            opt.step()
            opt.zero_grad()
        #print(accuracy(model(xb), yb))


def fit_EM(x_train, y_train, x_nolabel, model, opt, loss1, loss2, Beta, epochs, bs1, bs2):
    n = len(x_train)
    for epoch in range(epochs):
        for i in range((n - 1) // bs1 + 1):
            # print('Epoch: {}, Batch:{}'.format(epoch, i))
            start_i = i * bs1
            end_i = start_i + bs1
            xbL = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xbL)

            start_i = i * bs2
            end_i = start_i + bs2
            xbU = x_nolabel[start_i:end_i]
            predU = model(xbU)
            pred2 = F.softmax(predU, 1)

            loss = loss1(pred, yb) + Beta * loss2(pred2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
            opt.step()
            opt.zero_grad()


def fit_method(x_train, y_train, x_nolabel, M1, M2, M3, opt, loss1, loss2, Beta, epochs, bs1, bs2):
    n = len(x_train)
    for epoch in range(epochs):
        for i in range((n - 1) // bs1 + 1):
            start_i = i * bs1
            end_i = start_i + bs1
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            start_i = i * bs2
            end_i = start_i + bs2
            xU = x_nolabel[start_i:end_i]
            pred = M3(M1(xb))
            preds = F.softmax(pred, 1)
            pred1 = M3(M2(xU))
            pred2 = M3(M2(xb))
            pred1 = F.softmax(pred1, 1)
            loss = (1 - Beta) * (loss1(pred, yb) + loss1(pred2, yb)) + Beta * (loss2(torch.cat([pred1, preds], dim=0)))
            # print('Epoch:{}, Batch:{}, Loss:{}'.format(epoch, i, loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(M1.parameters(), 100)
            torch.nn.utils.clip_grad_norm_(M2.parameters(), 100)
            torch.nn.utils.clip_grad_norm_(M3.parameters(), 100)
            opt.step()
            opt.zero_grad()


model_1 = Mnist_CNN_1()
opt_L = optim.SGD(model_1.parameters(), lr=lr_L, momentum=0.9)
fit_L(train_dl_label, model_1, opt_L, loss_func_L, epochs_L)
print("Accuracy of a model trained only on labeled data:")
print(accuracy(model_1(x_test), y_test))

model_3 = Mnist_CNN_1()
opt_L = optim.SGD(model_3.parameters(), lr=lr_L, momentum=0.9)
fit_EM(x_train_label, y_train_label, x_train_nolabel, model_3, opt_L, loss_func_L, loss_func_UL,
                                beta, epochs_L, bs_L, bs_UL)
print("Accuracy of a EM with entropy of unlabelled only:")
print(accuracy(model_3(x_test), y_test))


M1 = Representation()
M2 = Representation()
M3 = output()
params = list(M1.parameters()) + list(M2.parameters()) + list(M3.parameters())
opt_method = optim.SGD(params, lr=lr_L, momentum=0.9)
fit_method(x_train_label, y_train_label, x_train_nolabel, M1, M2, M3, opt_method, loss_func_L,
           loss_func_UL, beta, epochs_L, bs_L, bs_UL)
print("Accuracy of our Method:")
print(accuracy(M3(M2(x_test)), y_test))


model_2 = Mnist_CNN_1()
opt_L = optim.SGD(model_2.parameters(), lr=lr_L, momentum=0.9)
fit_L(train_dl, model_2, opt_L, loss_func_L, epochs_L)
print("Accuracy of a model trained with labels of unlabelled data (Upper bound):")
print(accuracy(model_2(x_test), y_test))

