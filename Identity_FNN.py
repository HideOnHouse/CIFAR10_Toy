import random

import torch
import torch.nn
import torch.functional

from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class IdentityFNN(torch.nn.Module):
    def __init__(self):
        super(IdentityFNN, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(1, 256),
            torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class CustomDataSet(Dataset):
    def __init__(self, size):
        self.x = []
        self.t = []

        for i in range(size):
            temp = random.randint(0, 100000)
            self.x.append([temp])
            self.t.append([temp])

        self.x = torch.FloatTensor(self.x)
        self.t = torch.FloatTensor(self.t)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.t[item]


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)


def main():
    # Build model
    model = IdentityFNN()
    model.apply(init_weights)
    # Hyper Parameters
    learning_rate = 0.00001
    batch_size = 10
    epoch = 20

    # Make Train set and Test set
    train_set = CustomDataSet(10000)

    # Start Learning
    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for i in (range(epoch)):
        temp = 0
        for x, t in DataLoader(train_set, num_workers=2, batch_size=batch_size):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, t)
            loss.backward()
            optimizer.step()
            temp = loss.item()
        print("Epoch : %s, Loss : %s" % (i + 1, temp))

    # Evaluation Start
    while True:
        answer = int(input(">> "))
        pred = model(torch.FloatTensor([answer]))
        print("Your Input : %s, Model Output : %s" % (answer, pred.item()))


if __name__ == '__main__':
    main()
