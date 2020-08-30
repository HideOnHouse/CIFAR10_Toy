# torch import
import torch
import torch.nn as nn
import torch.utils.data
import torchvision

from tqdm import tqdm
from time import sleep


# Convolution 2D customized because of image size
class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 48, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer5 = torch.nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = torch.nn.Sequential(
            nn.Dropout(),
            nn.Linear(2304, 4608),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4608, 4608),
            nn.ReLU(inplace=True),
            nn.Linear(4608, 10),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# noinspection DuplicatedCode
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AlexNet()
    model.to(device)

    # Hyper Parameter
    epoch = 20
    batch_size = 100
    learning_rate = 0.001

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=True, transform=torchvision.transforms.ToTensor(), download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=True, transform=torchvision.transforms.ToTensor(), download=True),
        batch_size=1,
        drop_last=True,
        num_workers=2
    )

    # Train Sequence
    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    for i in range(epoch):
        temp = 0
        for data, label in tqdm(train_loader, bar_format='{l_bar}{bar:100}{r_bar}'):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            hypothesis = model(data)
            loss = criterion(hypothesis, label)
            loss.backward()
            optimizer.step()
            temp = loss.item()
        sleep(0.1)
        print("Epoch : %s, Loss : %s" % (i + 1, temp))

    torch.save(model, 'MNIST_Alex.pth')
    model = torch.load('MNIST_Alex.pth')

    # Test Sequence
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, label in tqdm(test_loader, bar_format='{l_bar}{bar:100}{r_bar}'):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(label.data).sum()

    print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()
