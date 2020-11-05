import torch
from matplotlib import pyplot as plt
from torch import nn as nn
import numpy as np
from tqdm import tqdm

SIZE = 20


class ReLUNetwork(nn.Module):
    def __init__(self):
        super(ReLUNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(1, SIZE),
            nn.ReLU(),
            nn.Linear(SIZE, SIZE),
            nn.ReLU(),
            nn.Linear(SIZE, 1)
        )

    def forward(self, x):
        out = self.features(x)
        return out


class SigmoidNetwork(nn.Module):
    def __init__(self):
        super(SigmoidNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(1, SIZE),
            nn.Sigmoid(),
            nn.Linear(SIZE, SIZE),
            nn.Sigmoid(),
            nn.Linear(SIZE, 1)
        )

    def forward(self, x):
        out = self.features(x)
        return out


class TanhNetwork(nn.Module):
    def __init__(self):
        super(TanhNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(1, SIZE),
            nn.Tanh(),
            nn.Linear(SIZE, SIZE),
            nn.Tanh(),
            nn.Linear(SIZE, 1)
        )

    def forward(self, x):
        out = self.features(x)
        return out


def f(x):
    return (x - 10) ** 3 + 7 * ((x - 10) ** 2) - 2 * (x - 10) + 15


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    activation = ["Sigmoid", "ReLU", "Tanh"]

    criterion = nn.MSELoss()

    models = [SigmoidNetwork(), ReLUNetwork(), TanhNetwork()]
    optimizers = []
    for model in models:
        optimizers.append(torch.optim.AdamW(model.parameters()))

    train_x = torch.tensor([[i] for i in np.arange(3, 13, 0.01)], dtype=torch.float)
    train_y = torch.tensor([[f(i.item())] for i in train_x], dtype=torch.float)

    test_x = torch.tensor([[i] for i in np.arange(3, 13, 0.01)], dtype=torch.float)
    test_y = torch.tensor([[f(i.item())] for i in test_x], dtype=torch.float)

    for model in models:
        model.train()
        model.to(device)

    criterion.to(device)
    train_x = train_x
    train_y = torch.log(train_y)
    with tqdm(range(5000), desc="Iteration: ") as t:
        for _ in t:
            loss = 0
            for i in range(3):
                answer = train_y.to(device)
                prediction = models[i](train_x.to(device))
                loss = criterion(prediction, answer)

                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()
            t.set_postfix(Loss='{:6f}'.format(loss.item()))

    plt.plot(test_x, test_y, label="Actual", linestyle='dashed')
    for model in models:
        model.eval()
    with torch.no_grad():
        for i, model in enumerate(models):
            model.cpu()
            plt.plot(test_x, torch.exp(model(test_x)).detach().numpy(), label=activation[i])
            plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("Regression Result.png")


if __name__ == '__main__':
    main()
