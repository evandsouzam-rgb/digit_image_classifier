import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def shift(X, seed, pixels=1, noise=0.15):
    rng = np.random.RandomState(seed)
    imgs = np.roll(X.reshape(-1, 8, 8), pixels, axis=2)
    imgs[:, :, :pixels] = 0
    imgs = imgs + rng.normal(0, noise, imgs.shape)
    return np.clip(imgs, 0, 1).astype(np.float32).reshape(-1, 64)


def make_loader(X, y, batch_size, shuffle=False):
    data = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def get_loaders(batch_size=32, seed=0, shifted=False, n_train=None):
    digits = load_digits()
    X = (digits.data / 16).astype(np.float32)
    y = digits.target.astype(np.int64)

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)

    if shifted:
        X_train = shift(X_train, seed)
        X_val = shift(X_val, seed + 1)
        X_test = shift(X_test, seed + 2)

    if n_train is not None and n_train < len(X_train):
        idx = np.random.RandomState(seed).choice(len(X_train), n_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    return (
        make_loader(X_train, y_train, batch_size, shuffle=True),
        make_loader(X_val, y_val, batch_size),
        make_loader(X_test, y_test, batch_size),
    )
