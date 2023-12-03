import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from PyXAB.algos.VROOM import VROOM

torch.manual_seed(1234)
random_idx = np.random.choice(np.arange(60000), 6000)
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
training_data.targets = training_data.targets[random_idx]
training_data.data = training_data.data[random_idx]

training_data, validation_data = random_split(training_data, [5000, 1000])
# random_idx_test = np.random.choice(np.arange(10000), 1000)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

device = ("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetwork(nn.Module):
    def __init__(self, depth):
        super().__init__()
        dim = [28 * 28] + [512 for i in range(depth)] + [10]
        self.flatten = nn.Flatten()
        self.linear_stack = nn.ModuleList([nn.Linear(dim[i - 1], dim[i]) for i in range(1, len(dim))])
        self.relu_stack = nn.ModuleList([nn.ReLU() for i in range(depth)])
        
    def forward(self, x):
        x = self.flatten(x)
        for i in range(len(self.linear_stack) - 1):
            x = self.linear_stack[i](x)
            x = self.relu_stack[i](x)
        return self.linear_stack[-1](x)

batch_size = 64
epochs = 1
depth = 1


loss_fn = nn.CrossEntropyLoss()

T = 100

domain = [[0.0001, 0.01]]
algo = VROOM(b=1, f_max=3, domain=domain)
for i in range(T):
    lr = algo.pull(i)[-1]
    model = NeuralNetwork(depth=depth)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('lr', lr)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n------------------------------")

        size = len(train_dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
                

        model.eval()
        # size = len(test_dataloader.dataset)
        num_batches = len(validation_dataloader)
        valid_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in validation_dataloader:
                pred = model(X)
                valid_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
        valid_loss /= num_batches
        print('valid loss:', valid_loss)
    algo.receive_reward(i, -valid_loss)    
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
print('final lr:', algo.get_last_point()[-1])
# test
lr = algo.get_last_point()[-1]
model = NeuralNetwork(depth=depth)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for t in range(10):
    print(f"Epoch {t + 1}\n------------------------------")

    size = len(train_dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            

    model.eval()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
