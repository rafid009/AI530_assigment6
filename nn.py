import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import time
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
print(f"Device Name: {torch.cuda.get_device_name(device) if device == 'cuda' else 'cpu'}")
print(f"Device Properties: {torch.cuda.get_device_properties(device) if device == 'cuda' else 'some cpu'}")



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NeuralNetwork2(NeuralNetwork):
    def __init__(self):
        super(NeuralNetwork2, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

class NeuralNetwork3(NeuralNetwork):
    def __init__(self):
        super(NeuralNetwork3, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

class NeuralNetwork4(nn.Module):
    def __init__(self):
        super(NeuralNetwork4, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

def prepare_datasets():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )   

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    return training_data, test_data

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct, mistakes = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            mistakes += (pred.argmax(1) != y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return mistakes / size


def calculate_gpu_time(train_loader, model, loss_fn, optimizer, epochs):
    train_loop(train_loader, model, loss_fn, optimizer)
    torch.cuda.synchronize() # wait for warm-up to finish
    times = []
    for e in range(epochs):
        start_epoch = time.time()
        train_loop(train_loader, model, loss_fn, optimizer)
        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        times.append(elapsed)

    total_time = sum(times)
    print(f"GPU time took: {total_time}")
    return total_time



if __name__=="__main__":
    train_data, test_data = prepare_datasets()
    train_loader = DataLoader(train_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)
    lr = 1e-3
    batch_size = 64
    epochs = 10

    print(f"ARCH - 1")
    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    time1 = calculate_gpu_time(train_loader, model, loss_fn, optimizer, epochs)
    MCR1 = test_loop(test_loader, model, loss_fn)
    print(f"MCR = {MCR1}")

    print(f"ARCH - 2")
    model = NeuralNetwork2()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    time2 = calculate_gpu_time(train_loader, model, loss_fn, optimizer, epochs)
    MCR2 = test_loop(test_loader, model, loss_fn)
    print(f"MCR = {MCR2}")

    print(f"ARCH - 3")
    model = NeuralNetwork3()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    time3 = calculate_gpu_time(train_loader, model, loss_fn, optimizer, epochs)
    MCR3 = test_loop(test_loader, model, loss_fn)
    print(f"MCR = {MCR3}")

    print(f"ARCH - 4")
    model = NeuralNetwork4()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    time4 = calculate_gpu_time(train_loader, model, loss_fn, optimizer, epochs)
    MCR4 = test_loop(test_loader, model, loss_fn)
    print(f"MCR = {MCR4}")

    print(f"\n\nArch - 1, MCR = {MCR1}\n\nArch - 2, MCR = {MCR2}\n\nArch - 3, MCR = {MCR3}\n\nArch - 4, MCR = {MCR4}")