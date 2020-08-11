from tqdm import tqdm

# torch import
import torch
import torch.utils.data
import torchvision


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 24, 3),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(4),
            torch.nn.Conv2d(24, 48, 3),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(4)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(48, 10, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleCNN()
    model.to(device)

    # Hyper Parameter
    epoch = 5
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
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            hypothesis = model(data)
            loss = criterion(hypothesis, label)
            loss.backward()
            optimizer.step()
            temp = loss.item()
        print("Epoch : %s, Loss : %s" % (i + 1, temp))

    # Test Sequence
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(label.data).sum()

    print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()
