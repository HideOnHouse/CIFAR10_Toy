# torch import
import torch
import torch.utils.data
import torchvision

from tqdm import tqdm
from time import sleep


# Convolution 2D customized because of image size
class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, 1, 1),
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.Conv2d(128, 128, 3, 1, 1),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, 1, 1),
            torch.nn.Conv2d(256, 256, 3, 1, 1),
            torch.nn.Conv2d(256, 256, 3, 1, 1),
            torch.nn.Conv2d(256, 512, 3, 1, 1),
            torch.nn.Conv2d(512, 512, 3, 1, 1),
            torch.nn.Conv2d(512, 512, 3, 1, 1),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(512, 512, 3, 1, 1),
            torch.nn.Conv2d(512, 512, 3, 1, 1),
            torch.nn.MaxPool2d(2, 2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# noinspection DuplicatedCode
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGG16()
    model.to(device)

    # Hyper Parameter
    epoch = 10
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
        batch_size=15000,
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

    torch.save(model, 'MNIST_VGG.pth')
    model = torch.load('MNIST_VGG.pth')

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

    print('Test set Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()
