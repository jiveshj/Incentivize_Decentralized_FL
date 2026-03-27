"""
Data partitioning and dataset utilities.

Implements Dirichlet-based non-IID partitioning and the paper's
60/40 test distribution (60% local, 40% uniform).
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional, Dict
from collections import Counter
import gzip
import struct
import pandas as pd
import os 
from PIL import Image

class CelebASmilingCSV(Dataset):
    def __init__(self, csv_path, img_dir, split_csv_path=None, split=None, transform=None):
        """
        csv_path: attributes CSV (with columns: image_id, Smiling, ...)
        img_dir: folder with face images
        split_csv_path: optional CSV with partition info (train/val/test)
        split: one of ["train", "val", "test"] if using split_csv_path
        """
        self.transform = transform

        df = pd.read_csv(csv_path)

        if split_csv_path is not None and split is not None:
            split_df = pd.read_csv(split_csv_path)
            # assume split_df has columns: image_id, partition (0=train,1=val,2=test)
            merged = df.merge(split_df, on="image_id")
            if split == "train":
                merged = merged[merged["partition"] == 0]
            elif split == "val":
                merged = merged[merged["partition"] == 1]
            elif split == "test":
                merged = merged[merged["partition"] == 2]
            df = merged

        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])  # adjust column name if needed
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Kaggle often encodes attributes as -1/1; map to 0/1
        smile = row["Smiling"]   # adjust if header differs
        y = 1 if smile == 1 else 0

        return img, y

class EMNISTBalancedDataset(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        split = "train" if train else "test"
        images_path = os.path.join(
            root, "EMNIST", "raw",
            f"emnist-balanced-{split}-images-idx3-ubyte.gz"
        )
        labels_path = os.path.join(
            root, "EMNIST", "raw",
            f"emnist-balanced-{split}-labels-idx1-ubyte.gz"
        )

        self.data = self._read_image_file(images_path)   
        self.targets = self._read_label_file(labels_path)  

    def _read_label_file(self, path):
        with gzip.open(path, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            # magic for labels should be 2049
            if magic != 2049:
                raise RuntimeError(f"Invalid magic number in label file: {magic}")
            buf = f.read()
            labels = torch.frombuffer(buf, dtype=torch.uint8).long()
        return labels

    def _read_image_file(self, path):
        with gzip.open(path, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            # magic for images should be 2051
            if magic != 2051:
                raise RuntimeError(f"Invalid magic number in image file: {magic}")
            buf = f.read()
            data = torch.frombuffer(buf, dtype=torch.uint8)
            data = data.view(num, rows, cols)
        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])
        # rotate to match torchvision EMNIST orientation
        img = torch.rot90(img, k=3, dims=(0, 1))  # or (0,1) depending on your read
        img = torch.flip(img, dims=(1,))          # horizontal flip if needed

        img = img.unsqueeze(0).float() / 255.0
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# ---------------------------------------------------------------------------
# Dirichlet-based non-IID partitioning (as used in the paper)
# ---------------------------------------------------------------------------

def partition_dirichlet(dataset: Dataset, n_workers: int, alpha: float,
                        seed: int = 42) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Partition dataset using Dirichlet distribution (Acar et al. 2021).

    Each client gets data from all classes but with skewed proportions
    controlled by alpha. Lower alpha = more heterogeneous.

    The paper uses:
        - alpha=0.1 for FashionMNIST, CIFAR-10
        - alpha=0.05 for EMNIST

    We also ensure roughly equal dataset sizes across clients.

    Args:
        dataset: PyTorch dataset with (data, label) pairs
        n_workers: number of clients
        alpha: Dirichlet concentration parameter
        seed: random seed

    Returns:
        client_indices: list of index lists, one per client
        client_distributions: list of label distributions per client
    """
    rng = np.random.RandomState(seed)

    # Extract all labels
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])

    n_samples = len(targets)
    n_classes = len(np.unique(targets))
    samples_per_client = n_samples // n_workers

    # Group indices by class
    class_indices = {c: np.where(targets == c)[0] for c in range(n_classes)}
    for c in class_indices:
        rng.shuffle(class_indices[c])

    # Sample Dirichlet proportions for each client
    # Shape: (n_workers, n_classes)
    proportions = rng.dirichlet(alpha * np.ones(n_classes), size=n_workers)

    # Allocate samples to clients
    client_indices = [[] for _ in range(n_workers)]
    class_pointers = {c: 0 for c in range(n_classes)}

    for i in range(n_workers):
        budget = samples_per_client
        # Normalize proportions to sum to budget
        client_prop = proportions[i]
        client_counts = (client_prop * budget).astype(int)

        # Fix rounding to hit exact budget
        diff = budget - client_counts.sum()
        # Add remainder to largest classes
        top_classes = np.argsort(-client_prop)
        for j in range(abs(diff)):
            if diff > 0:
                client_counts[top_classes[j % n_classes]] += 1
            else:
                client_counts[top_classes[j % n_classes]] -= 1

        for c in range(n_classes):
            n_take = client_counts[c]
            ptr = class_pointers[c]
            available = len(class_indices[c])

            if ptr + n_take > available:
                # Wrap around with shuffling
                idx = np.concatenate([
                    class_indices[c][ptr:],
                    class_indices[c][:max(0, n_take - (available - ptr))]
                ])
                class_pointers[c] = (ptr + n_take) % available
            else:
                idx = class_indices[c][ptr:ptr + n_take]
                class_pointers[c] = ptr + n_take

            client_indices[i].extend(idx.tolist())

        rng.shuffle(client_indices[i])

    # Compute label distributions per client
    client_distributions = []
    for indices in client_indices:
        labels = targets[indices]
        dist = np.zeros(n_classes)
        for c in range(n_classes):
            dist[c] = np.sum(labels == c) / len(labels) if len(labels) > 0 else 0
        client_distributions.append(dist)

    return client_indices, client_distributions


# ---------------------------------------------------------------------------
# 60/40 test split (paper's evaluation setup)
# ---------------------------------------------------------------------------

def create_client_test_loaders(test_dataset: Dataset,
                               client_distributions: List[np.ndarray],
                               n_workers: int,
                               local_fraction: float = 0.6,
                               test_samples_per_client: int = 500,
                               batch_size: int = 256,
                               seed: int = 42) -> List[DataLoader]:
    """
    Create per-client test loaders with the paper's 60/40 split:
    - 60% matches client's local training distribution
    - 40% drawn from uniform label distribution

    This tests both local accuracy and cross-client generalization.

    Args:
        test_dataset: full test dataset
        client_distributions: label distributions per client from partitioning
        n_workers: number of clients
        local_fraction: fraction of test data matching local distribution (0.6)
        test_samples_per_client: how many test samples per client
        batch_size: for the test dataloader
        seed: random seed

    Returns:
        list of DataLoaders, one per client
    """
    rng = np.random.RandomState(seed + 1000)

    # Extract test labels
    if hasattr(test_dataset, 'targets'):
        test_targets = np.array(test_dataset.targets)
    elif hasattr(test_dataset, 'labels'):
        test_targets = np.array(test_dataset.labels)
    else:
        test_targets = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

    n_classes = len(np.unique(test_targets))
    test_class_indices = {c: np.where(test_targets == c)[0] for c in range(n_classes)}
    for c in test_class_indices:
        rng.shuffle(test_class_indices[c])

    uniform_dist = np.ones(n_classes) / n_classes

    test_loaders = []
    for i in range(n_workers):
        n_local = int(test_samples_per_client * local_fraction)
        n_uniform = test_samples_per_client - n_local

        client_indices = []

        # Local portion: sample according to client's training distribution
        local_dist = client_distributions[i]
        local_counts = (local_dist * n_local).astype(int)
        diff = n_local - local_counts.sum()
        top_classes = np.argsort(-local_dist)
        for j in range(abs(diff)):
            local_counts[top_classes[j % n_classes]] += (1 if diff > 0 else -1)

        for c in range(n_classes):
            n_take = local_counts[c]
            available = test_class_indices[c]
            chosen = rng.choice(available, size=min(n_take, len(available)),
                                replace=len(available) < n_take)
            client_indices.extend(chosen.tolist())

        # Uniform portion
        uniform_counts = (uniform_dist * n_uniform).astype(int)
        diff = n_uniform - uniform_counts.sum()
        for j in range(abs(diff)):
            uniform_counts[j % n_classes] += 1

        for c in range(n_classes):
            n_take = uniform_counts[c]
            available = test_class_indices[c]
            chosen = rng.choice(available, size=min(n_take, len(available)),
                                replace=len(available) < n_take)
            client_indices.extend(chosen.tolist())

        subset = Subset(test_dataset, client_indices)
        test_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=False))

    return test_loaders


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def get_dataset(name: str, data_dir: str = "./data") -> Tuple[Dataset, Dataset, int]:
    """
    Load a dataset with appropriate transforms.

    Returns:
        (train_dataset, test_dataset, num_classes)
    """
    if name == "fashionmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        train = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=transform)
        test = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=transform)
        return train, test, 10

    elif name == "emnist":
        transform = transforms.Compose([
        transforms.Normalize((0.1751,), (0.3332,)),
    ])

        train = EMNISTBalancedDataset(data_dir, train=True, transform=transform)
        test = EMNISTBalancedDataset(data_dir, train=False, transform=transform)
        return train, test, 47

    elif name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train = datasets.CIFAR10(data_dir, train=True, download=True,
                                  transform=transform_train)
        test = datasets.CIFAR10(data_dir, train=False, download=True,
                                 transform=transform_test)
        return train, test, 10

    elif name == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
        train = datasets.CIFAR100(data_dir, train=True, download=True,
                                   transform=transform_train)
        test = datasets.CIFAR100(data_dir, train=False, download=True,
                                  transform=transform_test)
        return train, test, 100

    elif name == "celeba":
        transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        base = os.path.join(data_dir, "celeba")  # data_dir/celeba

        csv_attr = os.path.join(base, "list_attr_celeba.csv")
        csv_split = os.path.join(base, "list_eval_partition.csv")
        img_dir = os.path.join(base, "img_align_celeba")

        train = CelebASmilingCSV(
            csv_path=csv_attr,
            img_dir=img_dir,
            split_csv_path=csv_split,
            split="train",
            transform=transform_train,
        )
        test = CelebASmilingCSV(
            csv_path=csv_attr,
            img_dir=img_dir,
            split_csv_path=csv_split,
            split="test",
            transform=transform_test,
        )
        return train, test, 2  # smiling / not smiling

    elif name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train = datasets.MNIST(data_dir, train=True, download=True,
                                transform=transform)
        test = datasets.MNIST(data_dir, train=False, download=True,
                               transform=transform)
        return train, test, 10

    else:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Available: fashionmnist, emnist, cifar10, cifar100, mnist, celeba")


def get_alpha_for_dataset(dataset_name: str) -> float:
    """Return the Dirichlet alpha used in the paper for each dataset."""
    alphas = {
        "fashionmnist": 0.1,
        "cifar10": 0.1,
        "cifar100": 0.1,  
        "emnist": 0.1,
        "mnist": 0.1,
        "celeba": 0.1,
    }
    return alphas.get(dataset_name, 0.1)


# ---------------------------------------------------------------------------
# Data heterogeneity analysis
# ---------------------------------------------------------------------------

def compute_kl_divergence(client_dist: np.ndarray,
                          global_dist: np.ndarray,
                          eps: float = 1e-10) -> float:
    """KL(client || global) divergence."""
    p = client_dist + eps
    q = global_dist + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_label_entropy(client_dist: np.ndarray,
                          eps: float = 1e-10) -> float:
    """Shannon entropy of a client's label distribution."""
    p = client_dist + eps
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def analyze_data_heterogeneity(client_distributions: List[np.ndarray]) -> dict:
    """
    Analyze data heterogeneity across clients.

    Returns dict with per-client KL divergence, entropy, etc.
    """
    dists = np.array(client_distributions)
    global_dist = dists.mean(axis=0)

    kl_divs = [compute_kl_divergence(d, global_dist) for d in dists]
    entropies = [compute_label_entropy(d) for d in dists]

    return {
        "global_distribution": global_dist.tolist(),
        "kl_divergences": kl_divs,
        "entropies": entropies,
        "avg_kl": float(np.mean(kl_divs)),
        "max_kl": float(np.max(kl_divs)),
        "avg_entropy": float(np.mean(entropies)),
    }


# ---------------------------------------------------------------------------
# Train / validation split per client
# ---------------------------------------------------------------------------

def split_client_train_val(client_indices: List[List[int]],
                           val_fraction: float = 0.15,
                           seed: int = 42) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Split each client's data indices into train and held-out validation sets.

    The validation set is used for:
    - Estimating ρ_i (solo training loss threshold)
    - Evaluating F_i(x) for dropout decisions

    Both must be on the SAME held-out data for a fair comparison
    (is collaboration better than solo for this client?).

    Args:
        client_indices: list of index lists per client
        val_fraction: fraction of each client's data to hold out
        seed: random seed

    Returns:
        (train_indices, val_indices) — both are lists of lists
    """
    rng = np.random.RandomState(seed + 2000)
    train_indices = []
    val_indices = []

    for indices in client_indices:
        indices = np.array(indices)
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_fraction))
        val_indices.append(indices[:n_val].tolist())
        train_indices.append(indices[n_val:].tolist())

    return train_indices, val_indices