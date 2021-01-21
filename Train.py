import datetime
import os
from sys import argv

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from Model import *

MODELS = {
    "FNN": FNN,
    "LeNet": LeNet,
    "AlexNet": AlexNet,
    "VGG16": VGG16,
    "ResNet34": ResNet34
}


def main(args):
    if len(args) == 1:
        print("Usage: python Train.py [model] [batch_size] [epoch]")
        print("Available model: FNN, LeNet, AlexNet, VGG16, ResNet32")
        exit(0)
    today = str(datetime.date.today()).replace("-", "")

    model = MODELS[args[1]]()
    batch_size = int(args[2])
    epoch = int(args[3])
    save_path = os.curdir + os.sep + f'{today}_{epoch}_{args[1]}.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1), inplace=True)
    ])
    train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                  persistent_workers=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    model.to(device)
    criterion.to(device)
    model.train()
    for e in range(1, epoch + 1):
        with tqdm(train_dataloader) as t:
            for image, label in t:
                image = image.to(device)
                label = label.to(device)
                prediction = model(image)
                loss = criterion(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_postfix(epoch="{} of {}".format(e, epoch), loss="{:5f}".format(loss.item()))

    with open(save_path, 'wb') as f:
        torch.save(model, f)

    print("model saved at {}".format(save_path))


if __name__ == '__main__':
    main(argv)
