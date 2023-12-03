import numpy as np
import torch
from torch import nn
from PyXAB.algos.VROOM import VROOM
import pandas as pd
from torch.utils.data import Dataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

torch.manual_seed(1234)
random_idx = np.random.choice(np.arange(60000), 6000)
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
training_data.targets = training_data.targets[random_idx]
training_data.data = training_data.data[random_idx]

training_data, validation_data = random_split(training_data, [5000, 1000])
# random_idx_test = np.random.choice(np.arange(10000), 1000)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
valid_loader = DataLoader(validation_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetwork(nn.Module):
    def __init__(self, depth):
        super().__init__()
        dim = [28*28] + [512 for i in range(depth)] + [10]
        self.flatten = nn.Flatten()
        self.linear_stack = nn.ModuleList([nn.Linear(dim[i - 1], dim[i]) for i in range(1, len(dim))])
        self.relu_stack = nn.ModuleList([nn.ReLU() for i in range(depth)])
        
    def forward(self, x):
        x = self.flatten(x)
        for i in range(len(self.linear_stack) - 1):
            x = self.linear_stack[i](x)
            x = self.relu_stack[i](x)
        return self.linear_stack[-1](x)

T = 100
N_train = 5000
lr = 0.001
loss_fn = nn.CrossEntropyLoss()
epochs = 1 # determines the reward function

depth_selected = []
test_acc = []
for _ in range(5):
    algo = VROOM(T, b=2, f_max=1, domain=[[1, 10]])
    for i in range(T):
        print(f"Round {i + 1}\n")
        depth = int(round(algo.pull(i)[-1]))
        model = NeuralNetwork(depth=depth).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print('depth', depth)
        for t in range(epochs):
            print(f"Epoch {t + 1}\n------------------------------")

            size = N_train
            model.train()
            for batch, (X, y) in enumerate(train_loader):
                pred = model(X.to(device))
                loss = loss_fn(pred, y.to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                if batch % 10 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
                    

            model.eval()
            # size = len(test_dataloader.dataset)
            num_batches = 64
            valid_loss, correct = 0, 0

            with torch.no_grad():
                for X, y in valid_loader:
                    pred = model(X.to(device))
                    valid_loss += loss_fn(pred, y.to(device)).item()
                    correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()
                
            valid_loss /= num_batches
            print('valid loss:', valid_loss)
        algo.receive_reward(i, -valid_loss)    
        # correct /= size
        # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print('final depth:', int(round(algo.get_last_point()[-1])))
    depth_selected.append(int(round(algo.get_last_point()[-1])))
    
    print(np.mean(depth_selected), np.std(depth_selected))
# raise ValueError
# test
    depth = int(round(algo.get_last_point()[-1]))
    model = NeuralNetwork(depth=depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for t in range(100):
        print(f"Epoch {t + 1}\n------------------------------")

        size = N_train
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            pred = model(X.to(device))
            loss = loss_fn(pred, y.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
                

    model.eval()
    size = len(test_loader.dataset)
    num_batches = 64
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()
        
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    test_acc.append(100 * correct)

depth_selected = np.array(depth_selected)
test_acc = np.array(test_acc)
np.save("result/MNIST_our_depth", depth_selected)
np.save("result/MNIST_our_acc", test_acc)