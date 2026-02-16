"""
Decentralized SGD with Local Updates (Algorithm 1)
===================================================
Each worker maintains its own model, performs τ local SGD steps,
then averages with neighbors via a gossip mixing matrix W.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import argparse
from typing import List, Optional, Callable


# ---------------------------------------------------------------------------
# Topology helpers – generate common mixing matrices
# ---------------------------------------------------------------------------

def ring_topology(n: int) -> torch.Tensor:
    """Doubly-stochastic mixing matrix for a ring graph."""
    W = torch.zeros(n, n)
    for i in range(n):
        W[i, i] = 1 / 3
        W[i, (i - 1) % n] = 1 / 3
        W[i, (i + 1) % n] = 1 / 3
    return W


def fully_connected_topology(n: int) -> torch.Tensor:
    """Doubly-stochastic mixing matrix where every node is connected."""
    return torch.ones(n, n) / n


def grid_topology(n: int) -> torch.Tensor:
    """
    Doubly-stochastic Metropolis-Hastings mixing matrix for a 2-D grid.
    n must be a perfect square.
    """
    side = int(np.sqrt(n))
    assert side * side == n, "n must be a perfect square for grid topology"
    W = torch.zeros(n, n)
    for i in range(n):
        r, c = divmod(i, side)
        neighbors = []
        if r > 0:
            neighbors.append((r - 1) * side + c)
        if r < side - 1:
            neighbors.append((r + 1) * side + c)
        if c > 0:
            neighbors.append(r * side + c - 1)
        if c < side - 1:
            neighbors.append(r * side + c + 1)
        for j in neighbors:
            deg_i = len(neighbors) + 1  # include self
            # neighbor count for j
            rj, cj = divmod(j, side)
            deg_j = 1  # self
            if rj > 0: deg_j += 1
            if rj < side - 1: deg_j += 1
            if cj > 0: deg_j += 1
            if cj < side - 1: deg_j += 1
            W[i, j] = 1.0 / max(deg_i, deg_j)
        W[i, i] = 1.0 - W[i].sum()
    return W


TOPOLOGIES = {
    "ring": ring_topology,
    "fully_connected": fully_connected_topology,
    "grid": grid_topology,
}


# ---------------------------------------------------------------------------
# Helper: flatten / unflatten model parameters
# ---------------------------------------------------------------------------

def params_to_vector(model: nn.Module) -> torch.Tensor:
    """Flatten all parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def vector_to_params(vec: torch.Tensor, model: nn.Module):
    """Load a flat vector back into model parameters."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vec[offset: offset + numel].view_as(p))
        offset += numel


# ---------------------------------------------------------------------------
# Data partitioning (IID and non-IID)
# ---------------------------------------------------------------------------

def partition_data_iid(dataset, n_workers: int) -> List[Subset]:
    """Randomly split dataset into n_workers equal IID shards."""
    indices = np.random.permutation(len(dataset))
    shards = np.array_split(indices, n_workers)
    return [Subset(dataset, shard.tolist()) for shard in shards]


def partition_data_noniid(dataset, n_workers: int, n_shards_per_worker: int = 2) -> List[Subset]:
    """
    Non-IID split: sort by label, divide into shards, assign
    `n_shards_per_worker` shards to each worker.
    """
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    sorted_indices = np.argsort(targets)
    total_shards = n_workers * n_shards_per_worker
    shard_size = len(dataset) // total_shards
    shards = [sorted_indices[i * shard_size: (i + 1) * shard_size].tolist()
              for i in range(total_shards)]
    np.random.shuffle(shards)
    worker_subsets = []
    for i in range(n_workers):
        idx = []
        for s in range(n_shards_per_worker):
            idx.extend(shards[i * n_shards_per_worker + s])
        worker_subsets.append(Subset(dataset, idx))
    return worker_subsets


# ---------------------------------------------------------------------------
# Simple models for benchmarking
# ---------------------------------------------------------------------------

class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


class SimpleCNN(nn.Module):
    """Small CNN for CIFAR-10 / MNIST style tasks."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Core: Decentralized SGD with Local Updates
# ---------------------------------------------------------------------------

class DecentralizedSGD:
    """
    Algorithm 1 – Decentralized SGD with Local Updates.

    Parameters
    ----------
    model_fn : callable
        A function that returns a fresh nn.Module (the model architecture).
    n_workers : int
        Number of decentralized workers / nodes.
    mixing_matrix : torch.Tensor
        Doubly-stochastic N×N mixing matrix W.
    tau : int
        Number of local SGD steps between gossip rounds.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        n_workers: int,
        mixing_matrix: torch.Tensor,
        tau: int = 1,
        device: str = "cpu",
    ):
        self.n_workers = n_workers
        self.W = mixing_matrix.to(device)
        self.tau = tau
        self.device = device

        # Initialize each worker's local model (same init for fair comparison)
        init_model = model_fn().to(device)
        init_vec = params_to_vector(init_model)
        self.models: List[nn.Module] = []
        for _ in range(n_workers):
            m = model_fn().to(device)
            vector_to_params(init_vec.clone(), m)
            self.models.append(m)

    # ---- one communication round ----
    def train_round(
        self,
        loaders: List[DataLoader],
        lr: float,
        criterion: nn.Module,
        loader_iters: Optional[List] = None,
    ):
        """
        Perform one communication round:
          1. Each worker does τ local SGD steps.
          2. Gossip averaging via mixing matrix W.

        Returns
        -------
        avg_loss : float
            Average training loss across workers for this round.
        loader_iters : list
            Updated dataloader iterators (to maintain position across rounds).
        """
        if loader_iters is None:
            loader_iters = [iter(dl) for dl in loaders]

        round_loss = 0.0
        n_samples = 0

        # --- Step 1: τ local SGD updates per worker ---
        for i in range(self.n_workers):
            self.models[i].train()
            for _r in range(self.tau):
                # Sample a mini-batch (cycle through data if exhausted)
                try:
                    data, target = next(loader_iters[i])
                except StopIteration:
                    loader_iters[i] = iter(loaders[i])
                    data, target = next(loader_iters[i])

                data, target = data.to(self.device), target.to(self.device)

                # Forward + backward
                output = self.models[i](data)
                loss = criterion(output, target)
                self.models[i].zero_grad()
                loss.backward()

                # Local SGD update: x_i := x_i - η * ∇F_i
                with torch.no_grad():
                    for p in self.models[i].parameters():
                        p.data -= lr * p.grad

                round_loss += loss.item() * data.size(0)
                n_samples += data.size(0)

        # --- Step 2: Gossip averaging  x_i^{t+1,0} = Σ_j w_ij x_j^{t,τ} ---
        # Collect all parameter vectors
        all_vecs = torch.stack(
            [params_to_vector(m) for m in self.models]
        )  # shape (N, d)

        # Multiply by mixing matrix: new_vecs = W @ all_vecs
        new_vecs = self.W @ all_vecs  # (N, d)

        # Write back
        for i in range(self.n_workers):
            vector_to_params(new_vecs[i], self.models[i])

        avg_loss = round_loss / max(n_samples, 1)
        return avg_loss, loader_iters

    # ---- evaluation ----
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader, criterion: nn.Module):
        """
        Evaluate the *average model* on a test set.

        Returns (avg_loss, accuracy).
        """
        # Build average model
        avg_vec = torch.stack(
            [params_to_vector(m) for m in self.models]
        ).mean(dim=0)
        eval_model = copy.deepcopy(self.models[0])
        vector_to_params(avg_vec, eval_model)
        eval_model.eval()

        total_loss, correct, total = 0.0, 0, 0
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = eval_model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def consensus_distance(self) -> float:
        """
        Measure consensus: average ‖x_i − x̄‖² across workers.
        Useful for tracking how well workers agree.
        """
        vecs = torch.stack([params_to_vector(m) for m in self.models])
        mean_vec = vecs.mean(dim=0)
        return ((vecs - mean_vec).norm(dim=1) ** 2).mean().item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_dsgd(
    model_fn: Callable[[], nn.Module],
    train_subsets: List[Subset],
    test_dataset,
    n_workers: int,
    mixing_matrix: torch.Tensor,
    tau: int = 1,
    T: int = 100,
    batch_size: int = 32,
    lr: float = 0.01,
    lr_schedule: Optional[str] = None,
    lr_decay_rate: float = 0.1,
    lr_decay_milestones: Optional[List[int]] = None,
    device: str = "cpu",
    eval_every: int = 10,
    verbose: bool = True,
):
    """
    Full training loop for Decentralized SGD with Local Updates.

    Parameters
    ----------
    model_fn          : returns a fresh nn.Module
    train_subsets     : list of Subset, one per worker
    test_dataset      : test set (single)
    n_workers         : N
    mixing_matrix     : W (N×N doubly-stochastic)
    tau               : local steps between gossip
    T                 : total communication rounds
    batch_size        : mini-batch size per worker
    lr                : initial learning rate
    lr_schedule       : None | 'step' | 'cosine'
    lr_decay_rate     : γ for step schedule
    lr_decay_milestones : list of round indices to decay lr
    device            : 'cpu' or 'cuda'
    eval_every        : evaluate every this many rounds
    verbose           : print progress

    Returns
    -------
    history : dict with keys 'train_loss', 'test_loss', 'test_acc', 'consensus'
    dsgd    : the DecentralizedSGD object (contains final models)
    """
    # Build data loaders
    train_loaders = [
        DataLoader(sub, batch_size=batch_size, shuffle=True, drop_last=False)
        for sub in train_subsets
    ]
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    dsgd = DecentralizedSGD(
        model_fn=model_fn,
        n_workers=n_workers,
        mixing_matrix=mixing_matrix,
        tau=tau,
        device=device,
    )

    if lr_decay_milestones is None:
        lr_decay_milestones = []

    history = {"train_loss": [], "test_loss": [], "test_acc": [], "consensus": []}
    loader_iters = None

    for t in range(T):
        # --- learning rate schedule ---
        current_lr = lr
        if lr_schedule == "step":
            for ms in lr_decay_milestones:
                if t >= ms:
                    current_lr *= lr_decay_rate
        elif lr_schedule == "cosine":
            current_lr = lr * 0.5 * (1 + np.cos(np.pi * t / T))

        # --- one communication round ---
        train_loss, loader_iters = dsgd.train_round(
            train_loaders, current_lr, criterion, loader_iters
        )
        history["train_loss"].append(train_loss)

        # --- evaluation ---
        if (t + 1) % eval_every == 0 or t == T - 1:
            test_loss, test_acc = dsgd.evaluate(test_loader, criterion)
            consensus = dsgd.consensus_distance()
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)
            history["consensus"].append(consensus)

            if verbose:
                print(
                    f"Round {t+1:>4d}/{T} | "
                    f"lr={current_lr:.5f} | "
                    f"train_loss={train_loss:.4f} | "
                    f"test_loss={test_loss:.4f} | "
                    f"test_acc={test_acc*100:.2f}% | "
                    f"consensus={consensus:.6f}"
                )

    return history, dsgd


# ---------------------------------------------------------------------------
# Example: MNIST with ring topology
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Decentralized SGD with Local Updates")
    parser.add_argument("--n_workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--tau", type=int, default=4, help="Local steps between gossip")
    parser.add_argument("--T", type=int, default=200, help="Communication rounds")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["none", "step", "cosine"])
    parser.add_argument("--topology", type=str, default="ring",
                        choices=list(TOPOLOGIES.keys()))
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "cifar10"])
    parser.add_argument("--partition", type=str, default="iid",
                        choices=["iid", "noniid"])
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Dataset
    from torchvision import datasets, transforms

    if args.dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
        model_fn = lambda: SimpleCNN(in_channels=1, num_classes=10)

    elif args.dataset == "cifar10":
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
        train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
        # Adjust CNN for 3-channel 32×32 images
        model_fn = lambda: SimpleCNN(in_channels=3, num_classes=10)

    # Partition data across workers
    if args.partition == "iid":
        train_subsets = partition_data_iid(train_dataset, args.n_workers)
    else:
        train_subsets = partition_data_noniid(train_dataset, args.n_workers)

    # Mixing matrix
    W = TOPOLOGIES[args.topology](args.n_workers)
    print(f"\n{'='*60}")
    print(f"Decentralized SGD with Local Updates")
    print(f"{'='*60}")
    print(f"  Workers     : {args.n_workers}")
    print(f"  Topology    : {args.topology}")
    print(f"  τ (local)   : {args.tau}")
    print(f"  Rounds (T)  : {args.T}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr} ({args.lr_schedule})")
    print(f"  Dataset     : {args.dataset} ({args.partition})")
    print(f"  Device      : {args.device}")
    print(f"  Spectral gap: {1 - torch.linalg.eigvalsh(W)[-2].item():.4f}")
    print(f"{'='*60}\n")

    schedule = None if args.lr_schedule == "none" else args.lr_schedule

    history, dsgd = train_dsgd(
        model_fn=model_fn,
        train_subsets=train_subsets,
        test_dataset=test_dataset,
        n_workers=args.n_workers,
        mixing_matrix=W,
        tau=args.tau,
        T=args.T,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_schedule=schedule,
        device=args.device,
        eval_every=args.eval_every,
        verbose=True,
    )

    print(f"\nFinal test accuracy: {history['test_acc'][-1]*100:.2f}%")
    print(f"Final consensus distance: {history['consensus'][-1]:.6f}")

    # Save results
    import json
    results = {
        "args": vars(args),
        "train_loss": history["train_loss"],
        "test_loss": history["test_loss"],
        "test_acc": history["test_acc"],
        "consensus": history["consensus"],
    }
    with open("dsgd_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to dsgd_results.json")


if __name__ == "__main__":
    main()