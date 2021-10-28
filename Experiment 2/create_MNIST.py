import random
from pathlib import Path
import requests
import pickle
import gzip
import torch
from torchvision import datasets, transforms

random.seed(1000)
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

x_train = x_train[0:40000, :]
y_train = y_train[0:40000]

x_test = x_valid[0:6000, :]
y_test = y_valid[0:6000]


def MNIST_data(p=0.9):
    mnist_train_label = []
    mnist_train_nolabel = []
    x_train_label = []
    y_train_label = []

    x_train_nolabel = []
    y_train_nolabel = []

    count0 = 0
    count1 = 0
    for a, b in zip(x_train, y_train):
        if b <= 4:
            if count0 < 20000 * p:
                mnist_train_label.append((a, b))
                count0 += 1
                x_train_label.append(a)
                y_train_label.append(b)
            else:
                mnist_train_nolabel.append((a, b))
                x_train_nolabel.append(a)
                y_train_nolabel.append(b)
        if b >= 5:
            if count1 < 20000 * p:
                mnist_train_nolabel.append((a, b))
                count1 += 1
                x_train_nolabel.append(a)
                y_train_nolabel.append(b)
            else:
                mnist_train_label.append((a, b))
                x_train_label.append(a)
                y_train_label.append(b)

    mnist_test_label = []
    mnist_test_nolabel = []

    x_test_nolabel = []
    y_test_nolabel = []

    count0 = 0
    count1 = 0
    for a, b in zip(x_test, y_test):
        if b <= 4:
            if count0 < 3000 * p:
                mnist_test_label.append((a, b))
                count0 += 1
            else:
                mnist_test_nolabel.append((a, b))
                x_test_nolabel.append(a)
                y_test_nolabel.append(b)
        if b >= 5:
            if count1 < 3000 * p:
                mnist_test_nolabel.append((a, b))
                count1 += 1
                x_test_nolabel.append(a)
                y_test_nolabel.append(b)
            else:
                count1 += 1
                mnist_test_label.append((a, b))

    c1 = list(zip(x_train_label, y_train_label))
    random.shuffle(c1)
    x_train_label, y_train_label = zip(*c1)
    c2 = list(zip(x_train_nolabel, y_train_nolabel))
    random.shuffle(c2)
    x_train_nolabel, y_train_nolabel = zip(*c2)
    c3 = list(zip(x_test_nolabel, y_test_nolabel))
    random.shuffle(c3)
    x_test_nolabel, y_test_nolabel = zip(*c3)
    return torch.FloatTensor(x_train_label), torch.LongTensor(y_train_label), torch.FloatTensor(x_train_nolabel), \
           torch.LongTensor(y_train_nolabel), torch.FloatTensor(x_test_nolabel), torch.LongTensor(y_test_nolabel)
