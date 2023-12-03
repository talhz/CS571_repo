import numpy as np
import torch
from torch import nn
from PyXAB.algos.VROOM import VROOM
import pandas as pd

from utils.spiral import load_data_spiral

from bayes_opt import BayesianOptimization


torch.manual_seed(123)


device = ("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetwork(nn.Module):
    def __init__(self, depth):
        super().__init__()
        dim = [2] + [32 for i in range(depth)] + [2]
        self.flatten = nn.Flatten()
        self.linear_stack = nn.ModuleList([nn.Linear(dim[i - 1], dim[i]) for i in range(1, len(dim))])
        self.relu_stack = nn.ModuleList([nn.ReLU() for i in range(depth)])
        
    def forward(self, x):
        x = self.flatten(x)
        for i in range(len(self.linear_stack) - 1):
            x = self.linear_stack[i](x)
            x = self.relu_stack[i](x)
        return self.linear_stack[-1](x)


loss_fn = nn.CrossEntropyLoss()


def nn_learning(depth):
    batch_size = 64
    epochs = 10
    model = NeuralNetwork(depth=int(round(depth))).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    #print('lr', lr)
    for t in range(epochs):
        #print(f"Epoch {t + 1}\n------------------------------")

        size = len(train_loader.dataset)
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            pred = model(X.to(device))
            loss = loss_fn(pred, y.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 10 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                #print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
                

        model.eval()
        # size = len(test_dataloader.dataset)
        num_batches = len(valid_loader)
        valid_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in valid_loader:
                pred = model(X.to(device))
                valid_loss += loss_fn(pred, y.to(device)).item()
                correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()
            
        valid_loss /= num_batches
        #print('valid loss:', valid_loss)
    return -valid_loss
        # correct /= size
        # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# test
def test(depth):
    epochs = 100
    model = NeuralNetwork(depth=int(round(depth))).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for t in range(epochs):
        #print(f"Epoch {t + 1}\n------------------------------")

        size = len(train_loader.dataset)
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            pred = model(X.to(device))
            loss = loss_fn(pred, y.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                #print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
                

    model.eval()
    size = len(test_loader)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()
        
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct
    
        
domain = {'depth': (1, 10)}
result_depth = {}
result_acc = {}

for omega in range(0, 21, 2):
    depth_selected = []
    test_acc = []
    train_loader, valid_loader, test_loader = load_data_spiral(omega, 64, 1234)
    N_train = len(train_loader.sampler)
    for iter in range(5):
        print("current iter: ", iter)
        optimizer = BayesianOptimization(f=nn_learning, pbounds=domain,random_state=1,allow_duplicate_points=True)
        optimizer.maximize(init_points=50, n_iter=100)
        best_depth=optimizer.max['params']['depth']
        depth_selected.append(int(round(best_depth)))
        loss,acc=test(best_depth)
        test_acc.append(acc)
    result_depth[str(omega)] = depth_selected
    result_acc[str(omega)] = test_acc
    
result_depth = pd.DataFrame.from_dict(result_depth)
result_acc = pd.DataFrame.from_dict(result_acc)
result_depth.to_csv("result/spiral_GP_depth.csv")
result_acc.to_csv("result/spiral_GP_acc.csv")





