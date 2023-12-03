import numpy as np
import torch

def generate_spiral(phase: float, n: int = 1000, seed: int = 0):
    """Generate a spiral dataset

    Parameters
    ----------
    phase: float
        Phase of the spiral.
    n: int
        Number of samples.
    seed: int
        Random seed.

    Returns
    -------
    labels: list of int
        Labels of each each generated point, corresponding to which arm of the spiral the point is from.
    xy: array like
        Coordinates of the generated points

    """
    omega = lambda x: phase * np.pi / 2 * np.abs(x)
    rng = np.random.default_rng(seed)
    ts = rng.uniform(-1, 1, n)
    ts = np.sign(ts) * np.sqrt(np.abs(ts))
    xy = np.array([ts * np.cos(omega(ts)), ts * np.sin(omega(ts))]).T
    xy = rng.normal(xy, 0.02)
    labels = (ts >= 0).astype(int)
    return labels, xy


def load_data_spiral(phase: float, batch_size: int, seed: int = 0):
    """Build the dataloaders for the spiral dataset.
    The spiral datasets are generated and then wrapped in a dataloader.

    Parameters
    ----------
    phase: float
        Phase of the spiral.
    batch_size: int
        Batch size of the dataloaders.
    seed: int
        Random seed to use to generate the spiral data.

    Returns
    -------
    train_loader: DataLoader
        DataLoader for the training points of the spiral dataset.
    valid_loader: DataLoader
        DataLoader for the validation points of the spiral dataset.
    train_loader: DataLoader
        test_loader for the testing points of the spiral dataset.

    """
    t, xy = generate_spiral(phase, n=1024, seed=0 + seed)
    t_val, xy_val = generate_spiral(phase, n=1024, seed=1 + seed)
    t_test, xy_test = generate_spiral(phase, n=1024, seed=2 + seed)

    torch.manual_seed(seed)

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(xy), torch.LongTensor(t))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(xy_val), torch.LongTensor(t_val))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(xy_test), torch.LongTensor(t_test))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader