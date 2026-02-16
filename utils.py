"""
Utility functions: parameter flattening, metrics, seed management.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import os
from datetime import datetime
from typing import Dict, List, Any


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Parameter vector <-> model conversion
# ---------------------------------------------------------------------------

def params_to_vector(model: nn.Module) -> torch.Tensor:
    """Flatten all parameters into a single contiguous vector."""
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def vector_to_params(vec: torch.Tensor, model: nn.Module):
    """Load a flat vector back into model parameters (in-place)."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vec[offset: offset + numel].reshape_as(p))
        offset += numel


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader, criterion, device="cpu"):
    """
    Evaluate a single model on a dataloader.

    Returns:
        dict with 'loss', 'accuracy', 'correct', 'total'
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += criterion(output, target).item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
    }


@torch.no_grad()
def evaluate_per_client(models: List[nn.Module], test_loaders: List,
                        criterion, device="cpu"):
    """
    Evaluate each client's model on its own test loader.

    Returns:
        list of dicts, one per client
    """
    results = []
    for model, loader in zip(models, test_loaders):
        results.append(evaluate_model(model, loader, criterion, device))
    return results


# ---------------------------------------------------------------------------
# Consensus distance
# ---------------------------------------------------------------------------

@torch.no_grad()
def consensus_distance(models: List[nn.Module]) -> float:
    """Average ||x_i - x_bar||^2 across all workers."""
    vecs = torch.stack([params_to_vector(m) for m in models])
    mean_vec = vecs.mean(dim=0)
    return ((vecs - mean_vec).norm(dim=1) ** 2).mean().item()


# ---------------------------------------------------------------------------
# Results saving / loading
# ---------------------------------------------------------------------------

def save_results(results: Dict[str, Any], filepath: str):
    """Save results dict to JSON, converting numpy/tensor types."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(results, default=convert))
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {filepath}")


def generate_experiment_name(args) -> str:
    """Generate a descriptive experiment name from args."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (f"{args.algorithm}_{args.dataset}_{args.topology}_"
            f"N{args.n_workers}_tau{args.tau}_T{args.T}_{timestamp}")