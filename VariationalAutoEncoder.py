import datetime
import os

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.stats import norm
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),  # 1, 28, 28 --> 32, 26, 26
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),  # 32, 26, 26 --> 64, 24, 24
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),  # 64, 24, 24 --> 64, 22, 22
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),  # 64, 22, 22 --> 64, 20, 20
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(25600, 32),  # 64, 20, 20 --(25600)-> 32
            nn.ReLU(),
            nn.Linear(32, 2)  # 32 --> 2
        )

    def forward(self, x):
        x = self.layer1(x)
        x = torch.flatten(x, 1)
        out = self.layer2(x)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(2, 25600),  # 2 --> 12544
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1),  # 12544 --(64, 14, 14)-> 32, 28, 28
            nn.ReLU(),
            nn.Conv2d(32, 1, 3),  # 32, 28, 28  -->
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(-1, 64, 14, 14)
        out = self.layer2(x)
        return out


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)
        return out


def train(batch_size, epoch):
    today = str(datetime.date.today()).replace("-", "")
    save_path = os.curdir + os.sep + f'{today}_{epoch}_{"VAE"}.pth'

    model = VAE()
    train_dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters())

    model.to(DEVICE)
    criterion.to(DEVICE)
    model.train()
    for e in range(epoch):
        epoch_loss = 0
        count = 0
        with tqdm(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)) as t:
            for img, _ in t:
                img = img.to(DEVICE)
                prediction = model(img)
                loss = criterion(img, prediction)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                count += 1
                t.set_postfix(epoch=f"{e + 1} of {epoch}", loss=f"{epoch_loss / count:6f}")
    with open(save_path, 'wb') as f:
        torch.save(model, f)
    return model


def generate():
    models = [i for i in os.listdir(os.curdir) if i.split(os.extsep)[-1] == 'pth']
    for idx, i in enumerate(models):
        print(f"{idx + 1}: {i}")
    selected = int(input())
    with open(models[selected - 1], 'rb') as f:
        model = torch.load(f)

    model.to(DEVICE)
    model.eval()
    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.FloatTensor([[xi, yi]])
                z_sample = z_sample.to(DEVICE)
                # print(z_sample)
                x_decoded = model.decoder(z_sample)
                x_decoded = x_decoded.cpu().detach().numpy()
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10), dpi=300)
        plt.imshow(figure, cmap='Greys_r')
        plt.show()


if __name__ == '__main__':
    # train(1024, 64)
    generate()
