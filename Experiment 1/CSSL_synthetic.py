import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import CSSL_models
import CSSL_fit_functions
import copy
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import argparse

# import random

np.random.seed(111)
torch.manual_seed(111)

"""
np.random.seed(123)
torch.manual_seed(10)
"""
parser = argparse.ArgumentParser(description="paraameters for synthetic data, and CSSL")
parser.add_argument("--nL", default=400, type=int, help="number of labelled data points")
parser.add_argument("--nUL", default=4000, type=int, help="number of unlabelled data points")
parser.add_argument("--bs_L", default=64, type=int, help="batch size for labelled data")
parser.add_argument("--bs_UL", default=64, type=int, help="batch size for unlabelled data")
parser.add_argument("--epochs_L", default=20, type=int, help="epochs for labelled data")
parser.add_argument("--epochs_UL", default=10, type=int, help="epochs for unlabelled data")
parser.add_argument("--lr_L", default=0.08, type=float, help="learning rate for labelled data")
parser.add_argument("--lr_UL", default=0.08, type=float, help="learning rate for unlabelled data")
parser.add_argument("--a1", default=0.05, type=float, help="mean of Gaussians")
parser.add_argument("--a2", default=0.05, type=float, help="mean of Gaussians")
parser.add_argument("--s1", default=0.05, type=float, help="initial Variance of Gaussians")
parser.add_argument("--s2", default=0.05, type=float, help="Variance of Gaussians after shift")
parser.add_argument("--opt_L", default='SGD', type=str, help="labelled optimizer")
parser.add_argument("--opt_UL", default='SGD', type=str, help="unlabelled optimizer")
parser.add_argument("--Beta", default=0.1, type=float, help="hyperparamter for regularization term")

args = parser.parse_args()

bs_L = args.bs_L
lr_L = args.lr_L
epochs_L = args.epochs_L

bs_UL = args.bs_L * (int(args.nUL/args.nL))
if bs_UL == 0:
    bs_UL = 1
lr_UL = args.lr_UL
epochs_UL = args.epochs_UL


def syn_data_1(n=10000, a=0.05, s=0.05, m1=30, m2=20, c=0.5):
    x1 = a + s * np.random.randn(n, m1)
    x2 = -a + s * np.random.randn(n, m1)
    coef = (np.random.rand(n) > c).astype(int).reshape(-1, 1)
    x_c = coef * x1 + (1 - coef) * x2
    # temp = (np.sum(x_c, 1) > 0).astype(int)
    temp = np.sum(x_c, 1)
    z = 1 / (1 + np.exp(-temp))
    y = (np.random.rand(n) > z).astype(int)
    x_e = y.reshape(-1, 1) * 0.01 + 2 * np.random.randn(n, m2)
    x = np.concatenate((x_c, x_e), axis=1)

    return torch.FloatTensor(x), torch.LongTensor(y)


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def loss_func_L(inputL, target, eps=1e-8):
    CE = nn.CrossEntropyLoss()
    return CE(inputL, target)
    # return CE(input, target) + 1e-4 * torch.sum(-1 * (input+eps) * torch.log(input+eps))
    # return CE(inputL, target) + 1e-2 * (-inputL * torch.log(inputL + eps)).sum(1).mean()


def loss_func_UL(inputUL, eps=1e-8):
    return (-inputUL * torch.log(inputUL + eps)).sum(1).mean()
    # return torch.sum(-1 * inputUL * torch.log(inputUL+eps))


k = 5
acc_L = []
acc_Up = []
acc_EM_our = []
acc_EM_L = []
acc_Method = []

for i in range(k):

    x_train_label, y_train_label = syn_data_1(n=args.nL, a=args.a1, s=args.s1)

    x_train_nolabel, y_train_nolabel = syn_data_1(n=args.nUL, a=args.a2, s=args.s2)
    x_test, y_test = syn_data_1(n=args.nUL, a=args.a2, s=args.s2)

    train_ds_label = TensorDataset(x_train_label, y_train_label)
    train_dl_label = DataLoader(train_ds_label, batch_size=bs_L, shuffle=True)

    train_ds_nolabel = TensorDataset(x_train_nolabel, y_train_nolabel)
    train_dl_nolabel = DataLoader(train_ds_nolabel, batch_size=bs_UL, shuffle=True)



    if args.opt_UL == "Adam":
        opt_UL = optim.Adam(model1.parameters(), lr=0.01)
        print("hooray")
    else:
        opt_UL = optim.SGD(model1.parameters(), lr=lr_UL, momentum=0.9)

    model1 = CSSL_models.NN_model()
    opt1 = optim.SGD(model1.parameters(), lr=lr_L, momentum=0.9)

    CSSL_fit_functions.fit_L(train_dl_label, model1, opt1, loss_func_L, epochs_L)

    model2 = CSSL_models.NN_model()
    opt2 = optim.SGD(model2.parameters(), lr=lr_L, momentum=0.9)
    CSSL_fit_functions.fit_EM(x_train_label, y_train_label, x_train_nolabel, model2, opt2, loss_func_L, loss_func_UL,
                              args.Beta, epochs_L, bs_L, bs_UL)

    model3 = CSSL_models.NN_model()
    opt3 = optim.SGD(model3.parameters(), lr=lr_L, momentum=0.9)
    CSSL_fit_functions.fit_Upper(train_dl_label, train_dl_nolabel, model3, opt3, loss_func_L, epochs_L)

    model4 = CSSL_models.NN_model()
    opt4 = optim.SGD(model4.parameters(), lr=lr_L, momentum=0.9)
    CSSL_fit_functions.fit_EM_1(x_train_label, y_train_label, x_train_nolabel, model4, opt4, loss_func_L, loss_func_UL,
                                args.Beta, epochs_L, bs_L, bs_UL)

    M1 = CSSL_models.Representation()
    M2 = CSSL_models.Representation()
    M3 = CSSL_models.output()
    params = list(M1.parameters()) + list(M2.parameters()) + list(M3.parameters())
    opt5 = optim.SGD(params, lr=args.lr_L, momentum=0.9)
    CSSL_fit_functions.fit_Mahdi(x_train_label, y_train_label, x_train_nolabel, M1, M2, M3, opt5, loss_func_L,
                                 loss_func_UL, args.Beta, epochs_L, bs_L, bs_UL)


    acc_L.append(accuracy(model1(x_test), y_test))
    acc_Up.append(accuracy(model3(x_test), y_test))
    acc_EM_L.append(accuracy(model4(x_test), y_test))
    acc_EM_our.append(accuracy(model2(x_test), y_test))
    acc_Method.append(accuracy(M3(M2(x_test)), y_test))

acc_L = np.array(acc_L, dtype='float64')
acc_Up = np.array(acc_Up, dtype='float64')
acc_EM_our = np.array(acc_EM_our, dtype='float64')
acc_EM_L = np.array(acc_EM_L, dtype='float64')
acc_Method = np.array(acc_Method, dtype='float64')


print("\nAccuracy of a model trained only on labeled data:")
print('MSE = {} +/- {}'.format(np.mean(acc_L), np.std(acc_L)))

print("\nAccuracy of a EM with entropy of unlabelled only:")
print('MSE = {} +/- {}'.format(np.mean(acc_EM_L), np.std(acc_EM_L)))


print("\nAccuracy of our Method:")
print('MSE = {} +/- {}'.format(np.mean(acc_Method), np.std(acc_Method)))

print("\nAccuracy of a model trained with labels of unlabelled data (Upper bound):")
print('MSE = {} +/- {}'.format(np.mean(acc_Up), np.std(acc_Up)))

