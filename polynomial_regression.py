import torch
from matplotlib import pyplot as plt
from torch import nn as nn
import numpy as np
from tqdm import tqdm

SIZE = 100


class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(1, SIZE),
            nn.Tanh(),
            nn.Linear(SIZE, 1)
        )

    def forward(self, x):
        out = self.features(x)
        return out


def f(x):
    return 5 * (x ** 3) - 10 * (x ** 2) + 15 * x - 25


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.MSELoss()

    model = CustomNetwork()
    optimizer = torch.optim.AdamW(model.parameters())

    train_x = torch.tensor([[i] for i in np.arange(0, 13, 0.01)], dtype=torch.float)
    train_y = torch.tensor([[f(i.item())] for i in train_x], dtype=torch.float)

    test_x = torch.tensor([[i] for i in np.arange(-10, 20, 0.01)], dtype=torch.float)
    test_y = torch.tensor([[f(i.item())] for i in test_x], dtype=torch.float)

    model.train()
    model.to(device)

    criterion.to(device)
    train_x = train_x
    train_y = -torch.log((1 - train_y) ** 2)

    with tqdm(range(10000), desc="Iteration: ") as t:
        for _ in t:
            loss = 0
            for i in range(3):
                answer = train_y.to(device)
                prediction = model(train_x.to(device))
                loss = criterion(prediction, answer)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            t.set_postfix(Loss='{:6f}'.format(loss.item()))

    plt.plot(test_x, test_y, label="Actual", linestyle='dashed')
    model.cpu()
    plt.plot(test_x, -torch.exp(torch.sqrt(-model(test_x) + 1)).detach().numpy(), label="Predict")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
