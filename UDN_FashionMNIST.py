import argparse
import time

import numpy as np

import pandas as pd
import torch.optim as optim
import torch.utils.data

# from experiments.load_data import load_data_cifar
from utils.layer_generators import make_generators_fcn
from utils.models import UnboundedDepthNetwork, TruncatedPoisson, FixedDepth
from utils.train import train_one_epoch_classification

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

INPUT_SIZE = 784
OUTPUT_SIZE = 10
BATCH_SIZE = 64

CUDA = True
DEVICE = torch.device("cuda" if CUDA and torch.cuda.is_available() else "cpu")

print(DEVICE)


from torch.optim.lr_scheduler import _LRScheduler


class ExplicitLR(_LRScheduler):
    """"""

    def __init__(self, optimizer, lrs, last_epoch=-1, verbose=False):
        self.lrs = lrs
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= len(self.lrs):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.lrs[self.last_epoch] for _ in self.optimizer.param_groups]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-e", "--epochs", default=500, type=int, help="number of epochs")
    parser.add_argument("-l", "--layers", default=-1, help="number of layers", type=int)
    parser.add_argument("-s", "--seed", default=0, help="random seed", type=int)
    parser.add_argument("--categories", default="", help="filter categories", type=str)

    args = parser.parse_args()
    SEED = args.seed
    torch.manual_seed(SEED)
    L = args.layers

    device = DEVICE

    # LOAD DATA
    # Subsample categories to change the dataset complexity: [4,1] is (deer/car); [5,3] is (dog/cat)
    filter_labels = None
    if args.categories:
        filter_labels = list([int(c) for c in args.categories])

    # train_loader, valid_loader, test_loader = load_data_cifar(
    #     BATCH_SIZE, seed=SEED, filter_labels=filter_labels, validation_size=0
    # )
    # N_train = len(train_loader.sampler)
    torch.manual_seed(SEED)
    random_idx = np.random.choice(np.arange(60000), 6000)
    training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    training_data.targets = training_data.targets[random_idx]
    training_data.data = training_data.data[random_idx]

    training_data, validation_data = random_split(training_data, [5000, 1000])
    # random_idx_test = np.random.choice(np.arange(10000), 1000)
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    N_train = 5000

    generator_layers, generator_residual = make_generators_fcn(512, INPUT_SIZE, 10)
    # CREATE MODEL

   

    if L < 0:
        vpost = TruncatedPoisson(5.0)
    else:
        vpost = FixedDepth(L)

    model = UnboundedDepthNetwork(
        N_train,
        generator_layers,
        generator_residual,
        vpost,
        INPUT_SIZE,
        OUTPUT_SIZE,
        L_prior_poisson=1,
        theta_prior_scale=1.0,
    )

    model.model_name += ".cifar.v2" + ("-f" + args.categories if args.categories else "") + "-s%d" % SEED

    print(model.n_obs)
    model.set_device(device)

    # TRAINING LOOP
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = ExplicitLR(optimizer, [0.01] * 5 + [0.1] * 195 + [0.01] * 100 + [0.001] * 100)
    model.set_optimizer(optimizer)

    tmp = pd.DataFrame({"depth": [], "nu_L": [], "test_acc": []})
    tmp.to_csv("tmp.%s.csv" % model.model_name)

    for epoch in range(args.epochs):
        start_time = time.time()
        test_accuracy = train_one_epoch_classification(
            epoch,
            train_loader,
            valid_loader,
            test_loader,
            model,
            optimizer,
            scheduler,
            normalize_loss=True,
        )
        scheduler.step()
        print(time.time() - start_time)
