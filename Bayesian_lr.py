import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from bayes_opt import BayesianOptimization

torch.manual_seed(123)

torch.manual_seed(123)
random_idx = np.random.choice(np.arange(60000), 6000)
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
training_data.targets = training_data.targets[random_idx]
training_data.data = training_data.data[random_idx]

training_data, validation_data = random_split(training_data, [5000, 1000])
# random_idx_test = np.random.choice(np.arange(10000), 1000)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
# test_data.targets = test_data.targets[random_idx_test]
# test_data.data = test_data.data[random_idx_test]

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


loss_fn = nn.CrossEntropyLoss()


def nn_learning(lr):
    batch_size = 64
    epochs = 1
    depth = 1
    model = NeuralNetwork(depth=depth).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #print('lr', lr)
    for t in range(epochs):
        #print(f"Epoch {t + 1}\n------------------------------")

        size = len(train_dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
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
        num_batches = len(validation_dataloader)
        valid_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in validation_dataloader:
                pred = model(X.to(device))
                valid_loss += loss_fn(pred, y.to(device)).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
        valid_loss /= num_batches
        #print('valid loss:', valid_loss)
    return -valid_loss
        # correct /= size
        # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# test
def test(best_lr):
    epochs = 100
    depth = 1
    model = NeuralNetwork(depth=depth)
    optimizer = torch.optim.SGD(model.parameters(), lr=best_lr)
    for t in range(epochs):
        #print(f"Epoch {t + 1}\n------------------------------")

        size = len(train_dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            pred = model(X.to(device))
            loss = loss_fn(pred, y.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                #print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
                

        model.eval()
        size = len(test_dataloader.dataset)
        num_batches = len(test_dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in test_dataloader:
                pred = model(X.to(device))
                test_loss += loss_fn(pred, y.to(device)).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss,correct
    
def output(record):
    print(record)
    print(np.mean(record),np.std(record))
        
domain = {'lr': (0.0001, 0.01)}
lr_record=[]
loss_record=[]
acc_record=[]

for iter in range(5):
    print("current iter: ", iter)
    optimizer = BayesianOptimization(f=nn_learning, pbounds=domain,random_state=1,allow_duplicate_points=True)
    optimizer.maximize(init_points=50, n_iter=100)
    best_lr=optimizer.max['params']['lr']
    lr_record.append(best_lr)
    print('Learning Rate:', best_lr)  
    loss,acc=test(best_lr)
    loss_record.append(loss)
    acc_record.append(acc)
    output(lr_record)
    output(loss_record)
    output(acc_record)





