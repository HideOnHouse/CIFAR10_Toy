import torch
from torch import nn as nn
from torch.utils import data
import torchvision
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt


class ModerateCNN(nn.Module):
    def __init__(self):
        super(ModerateCNN, self).__init__()
        self.features = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )

        self.classifier = torch.nn.Sequential(
            nn.Linear(49152, 4096, bias=True),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(4096, 10, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out


def train(model, epoch=32, device='cuda'):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    batch_size = 1024

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('data', train=True, transform=torchvision.transforms.ToTensor(), download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        persistent_workers=True
    )

    model.to(device)
    criterion.to(device)
    for e in range(1, epoch + 1):
        with tqdm(train_loader, ascii=True) as t:
            for image, label in t:
                image = image.to(device)
                label = label.to(device)
                predict = model(image)
                loss = criterion(predict, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_postfix(loss=loss.item(), epoch="{} of {}".format(e, epoch))
    with open(f"CIFAR_10_Moderate.pt", 'wb') as f:
        torch.save(model, f)


def evaluate(model_path=None, device='cuda', show=False):
    dataset = torchvision.datasets.CIFAR10('data', train=True, transform=torchvision.transforms.ToTensor(),
                                           download=True)
    classes = dataset.classes
    if model_path is None:
        with open("CIFAR_10_Moderate.pt", 'rb') as f:
            model = torch.load(f)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        persistent_workers=True
    )

    # noinspection DuplicatedCode
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        with tqdm(test_loader, ascii=True) as t:
            count = 0
            for image, label in t:
                image = image.to(device)
                label = label.to(device)
                prediction = torch.argmax(model(image))
                if show:
                    plt.imshow(torchvision.transforms.ToPILImage()(image[0]), interpolation='bicubic')
                    plt.title(
                        "Prediction - {}, Label - {}".format(classes[prediction], classes[label.item()]))
                    plt.show()
                if label.item() == prediction:
                    count += 1
            print("Acc: {}".format(count / 10000))


def main(args):
    if args[1] == 'continue':
        with open("CIFAR_10_Moderate.pt", 'rb') as f:
            model = torch.load(f)
    else:
        model = ModerateCNN()
    if args[2] == 'train':
        if len(args) == 4:
            train(model, epoch=int(args[3]))
        else:
            train(model)
    else:
        if len(args) == 4:
            if args[3] == 'show':
                evaluate(show=True)
        else:
            evaluate()


if __name__ == '__main__':
    main(sys.argv)
