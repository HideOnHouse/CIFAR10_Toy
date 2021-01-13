import datetime
import os
from sys import argv

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from Model import *


def main(args):
    if len(args) == 1:
        print("Usage: python Train.py [model] [batch_size=1024] [epoch=16] [lr=1e-3] [save_path=./model.pth]")
        print("Available model: FNN, LeNet, AlexNet, VGG16")
        exit(0)
    today = str(datetime.date.today()).replace("-", "")

    model = AlexNet()
    batch_size = 1024
    epoch = 16
    lr = 1e-4
    save_path = os.curdir + os.sep + '{}_model.pth'.format(today)
    # batch_size = args[2]
    # epoch = args[3]
    # lr = args[4]
    # save_path = args[5]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                  persistent_workers=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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
                t.set_postfix(epoch="{} of {}".format(e, epoch), loss="{:6f}".format(loss.item()))

    with open(save_path, 'wb') as f:
        torch.save(model, f)

    print("model saved at {}".format(save_path))


if __name__ == '__main__':
    main(argv)
