"""
Core decentralized learning algorithms.

Algorithm 1: Decentralized SGD with Local Updates (baseline)
Algorithm 2: NodeDrop-IDSGD (incentivized, with pluggable weight strategies)

Both algorithms support:
- Multiple local SGD steps (τ)
- Gossip averaging via doubly-stochastic mixing matrix
- Client dropout simulation
- Per-client evaluation
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.data import DataLoader, Subset
from typing import List, Optional, Callable, Tuple

from utils import params_to_vector, vector_to_params, evaluate_model, consensus_distance


# ===========================================================================
# Algorithm 1: Decentralized SGD with Local Updates (Baseline)
# ===========================================================================

class DecentralizedSGD:
    """
    Algorithm 1 - Decentralized SGD with Local Updates.

    Standard baseline: all workers do τ local SGD steps then
    gossip-average with neighbors.
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

        # Initialize all workers with same parameters
        init_model = model_fn().to(device)
        init_vec = params_to_vector(init_model)
        self.models: List[nn.Module] = []
        for _ in range(n_workers):
            m = model_fn().to(device)
            vector_to_params(init_vec.clone(), m)
            self.models.append(m)

        # Track which clients are still active
        self.active = [True] * n_workers

    def train_round(
        self,
        loaders: List[DataLoader],
        lr: float,
        criterion: nn.Module,
        loader_iters: Optional[List] = None,
    ) -> Tuple[float, List]:
        """
        One communication round:
          1. Each active worker does τ local SGD steps
          2. Gossip averaging among active workers

        Returns (avg_train_loss, loader_iters)
        """
        if loader_iters is None:
            loader_iters = [iter(dl) if dl is not None else None for dl in loaders]

        round_loss = 0.0
        n_samples = 0

        # --- Local SGD updates ---
        for i in range(self.n_workers):
            if not self.active[i]:
                continue

            self.models[i].train()
            for _r in range(self.tau):
                try:
                    data, target = next(loader_iters[i])
                except (StopIteration, TypeError):
                    loader_iters[i] = iter(loaders[i])
                    data, target = next(loader_iters[i])

                data, target = data.to(self.device), target.to(self.device)
                output = self.models[i](data)
                loss = criterion(output, target)  #Computes average loss over the batch for this worker
                self.models[i].zero_grad()
                loss.backward()  #only compute gradients for this worker's model

                with torch.no_grad():
                    for p in self.models[i].parameters():
                        p.data -= lr * p.grad

                round_loss += loss.item() * data.size(0)
                n_samples += data.size(0)

        # --- Gossip averaging: X^{t+1,0} = X^{t,τ} @ W ---
        self._gossip_average()

        avg_loss = round_loss / max(n_samples, 1)
        return avg_loss, loader_iters

    def _gossip_average(self):
        """Apply gossip averaging among active workers using mixing matrix."""
        # Collect parameter vectors of all workers
        all_vecs = torch.stack([params_to_vector(m) for m in self.models])  # (N, d)

        # For inactive workers, their rows/cols in W should be zero
        # But we apply W as-is since inactive workers' vectors are frozen
        # Active workers average only with active neighbors via W
        new_vecs = self.W @ all_vecs  # (N, d)

        # Only update active workers
        for i in range(self.n_workers):
            if self.active[i]:
                vector_to_params(new_vecs[i], self.models[i])

    def get_average_model(self, active_only: bool = True) -> nn.Module:
        """Return the average model across (active) workers."""
        if active_only:
            active_indices = [i for i in range(self.n_workers) if self.active[i]]
        else:
            active_indices = list(range(self.n_workers))

        if len(active_indices) == 0:
            return copy.deepcopy(self.models[0])

        vecs = torch.stack([params_to_vector(self.models[i]) for i in active_indices])
        avg_vec = vecs.mean(dim=0)
        avg_model = copy.deepcopy(self.models[0])
        vector_to_params(avg_vec, avg_model)
        return avg_model


# ===========================================================================
# Algorithm 2: NodeDrop-IDSGD (Incentivized Decentralized SGD)
# ===========================================================================

class NodeDropIDSGD:
    """
    Algorithm 2 - NodeDrop-IDSGD.

    Extends Decentralized SGD with adaptive per-client learning rates
    based on dropout penalty weights a_i. The key difference from the
    baseline is the learning rate estimation phase at each round.

    The effective learning rate for client i is:
        η̂_i = η * b_i / (τ * (s_i + ε))

    where:
        b_i = (1-γ) + γ * a_i * q_i(x_i)     (pre-factor)
        q_i = σ_i * (1 - σ_i)                  (dropout gradient factor)
        σ_i = sigmoid(F_i(x_i) - ρ_i)          (dropout probability)
        s_i ≈ Σ_j b_j                          (scaling factor via gossip)
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        n_workers: int,
        mixing_matrix: torch.Tensor,
        importance_weights: np.ndarray,
        gamma: float = 0.5,
        epsilon: float = 1.0,
        tau: int = 1,
        tau_eta: int = 10,
        device: str = "cpu",
    ):
        """
        Args:
            model_fn: returns a fresh nn.Module
            n_workers: N
            mixing_matrix: doubly-stochastic W
            importance_weights: a_i for each node (from weight strategy)
            gamma: interpolation between ERM and dropout penalty [0, 1]
            epsilon: smoothing term in learning rate denominator
            tau: local SGD steps per round
            tau_eta: gossip steps for scaling factor estimation
            device: 'cpu' or 'cuda'
        """
        self.n_workers = n_workers
        self.W = mixing_matrix.to(device)
        self.a = torch.tensor(importance_weights, dtype=torch.float32, device=device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.tau_eta = tau_eta
        self.device = device

        # Client dropout thresholds ρ_i (estimated from local training)
        self.rho = torch.zeros(n_workers, device=device)
        self.rho_initialized = False

        # Initialize all workers with same parameters
        init_model = model_fn().to(device)
        init_vec = params_to_vector(init_model)
        self.models: List[nn.Module] = []
        for _ in range(n_workers):
            m = model_fn().to(device)
            vector_to_params(init_vec.clone(), m)
            self.models.append(m)

        self.active = [True] * n_workers

        # Logging
        self.dropout_probs = []
        self.effective_lrs = []

    def estimate_rho(self, loaders: List[DataLoader], criterion: nn.Module,
                     solo_rounds: int = 5, solo_lr: float = 0.01):
        """
        Estimate dropout thresholds ρ_i for each client.
        ρ_i = loss achieved by client i after a few rounds of local-only training.
        This serves as the client's reference for "what they could achieve alone".
        """
        for i in range(self.n_workers):
            # Create a temporary copy for solo training
            solo_model = copy.deepcopy(self.models[i])
            solo_model.train()
            loader_iter = iter(loaders[i])

            for _ in range(solo_rounds * self.tau):
                try:
                    data, target = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loaders[i])
                    data, target = next(loader_iter)

                data, target = data.to(self.device), target.to(self.device)
                output = solo_model(data)
                loss = criterion(output, target)
                solo_model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for p in solo_model.parameters():
                        p.data -= solo_lr * p.grad

            # Evaluate solo model loss as ρ_i
            solo_model.eval()
            total_loss, total_n = 0.0, 0
            loader_iter = iter(loaders[i])
            with torch.no_grad():
                for _ in range(min(5, len(loaders[i]))):
                    try:
                        data, target = next(loader_iter)
                    except StopIteration:
                        break
                    data, target = data.to(self.device), target.to(self.device)
                    output = solo_model(data)
                    total_loss += criterion(output, target).item() * data.size(0)
                    total_n += data.size(0)

            self.rho[i] = total_loss / max(total_n, 1)
            del solo_model

        self.rho_initialized = True

    def _compute_dropout_prob(self, client_idx: int, loss_val: float) -> float:
        """σ_i(x) = sigmoid(F_i(x) - ρ_i)"""
        return torch.sigmoid(torch.tensor(loss_val - self.rho[client_idx].item())).item()

    def _compute_prefactors(self, client_losses: List[float]) -> torch.Tensor:
        """
        Compute pre-factors b_i for all clients.
        b_i = (1 - γ) + γ * a_i * σ_i * (1 - σ_i)
        """
        b = torch.zeros(self.n_workers, device=self.device)
        sigma_vals = torch.zeros(self.n_workers, device=self.device)

        for i in range(self.n_workers):
            if not self.active[i]:
                continue
            sigma_i = self._compute_dropout_prob(i, client_losses[i])
            sigma_vals[i] = sigma_i
            q_i = sigma_i * (1 - sigma_i)
            b[i] = (1 - self.gamma) + self.gamma * self.a[i].item() * q_i

        return b, sigma_vals

    def _estimate_scaling_factors(self, b: torch.Tensor) -> torch.Tensor:
        """
        Estimate scaling factors s_i ≈ Σ_j b_j using τ_η gossip steps.

        Initialize: s_i^{0} = N * b_i
        Update: s^{r+1} = s^r @ W
        Final: s_i = max(s_i^{τ_η}, b_i)
        """
        # Initialize: each node starts with N * b_i
        s = self.n_workers * b.clone()  # (N,)

        # Gossip averaging for scaling factors (scalar gossip)
        for _ in range(self.tau_eta):
            s = self.W.T @ s  # s = W^T @ s (since W symmetric, same as W @ s)

        # Ensure non-degenerate: s_i >= b_i
        s = torch.max(s, b)

        return s

    def _get_client_losses(self, loaders: List[DataLoader],
                           criterion: nn.Module) -> List[float]:
        """Compute current empirical loss for each active client."""
        losses = [0.0] * self.n_workers
        for i in range(self.n_workers):
            if not self.active[i]:
                continue
            self.models[i].eval()
            total_loss, total_n = 0.0, 0
            with torch.no_grad():
                for data, target in loaders[i]:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.models[i](data)
                    total_loss += criterion(output, target).item() * data.size(0)
                    total_n += data.size(0)
                    # Only use a few batches for efficiency  (not completely sure about this but worth trying)
                    if total_n >= 256:
                        break
            losses[i] = total_loss / max(total_n, 1)
        return losses

    def train_round(
        self,
        loaders: List[DataLoader],
        base_lr: float,
        criterion: nn.Module,
        loader_iters: Optional[List] = None,
    ) -> Tuple[float, List, dict]:
        """
        One round of NodeDrop-IDSGD:
          1. Learning rate estimation phase (lines 3-7 of Algorithm 2)
          2. Local SGD with adaptive learning rates (lines 9-12)
          3. Gossip averaging (line 13)

        Returns:
            (avg_loss, loader_iters, round_info)
        """
        if loader_iters is None:
            loader_iters = [iter(dl) if dl is not None else None for dl in loaders]

        # --- Phase 1: Learning rate estimation ---
        client_losses = self._get_client_losses(loaders, criterion)
        b, sigma_vals = self._compute_prefactors(client_losses)
        s = self._estimate_scaling_factors(b)

        # Effective learning rate: η̂_i = η * b_i / (τ * (s_i + ε))
        effective_lrs = torch.zeros(self.n_workers, device=self.device)
        for i in range(self.n_workers):
            if self.active[i]:
                effective_lrs[i] = base_lr * b[i] / (self.tau * (s[i] + self.epsilon))

        # Store for logging
        self.dropout_probs.append(sigma_vals.cpu().numpy().copy())
        self.effective_lrs.append(effective_lrs.cpu().numpy().copy())

        # --- Phase 2: Local SGD with adaptive LR ---
        round_loss = 0.0
        n_samples = 0

        for i in range(self.n_workers):
            if not self.active[i]:
                continue

            lr_i = effective_lrs[i].item()
            self.models[i].train()

            for _r in range(self.tau):
                try:
                    data, target = next(loader_iters[i])
                except (StopIteration, TypeError):
                    loader_iters[i] = iter(loaders[i])
                    data, target = next(loader_iters[i])

                data, target = data.to(self.device), target.to(self.device)
                output = self.models[i](data)
                loss = criterion(output, target)
                self.models[i].zero_grad()
                loss.backward()

                with torch.no_grad():
                    for p in self.models[i].parameters():
                        p.data -= lr_i * p.grad

                round_loss += loss.item() * data.size(0)
                n_samples += data.size(0)

        # --- Phase 3: Gossip averaging ---
        self._gossip_average()

        avg_loss = round_loss / max(n_samples, 1)
        round_info = {
            "dropout_probs": sigma_vals.cpu().numpy(),
            "effective_lrs": effective_lrs.cpu().numpy(),
            "prefactors": b.cpu().numpy(),
            "scaling_factors": s.cpu().numpy(),
            "client_losses": client_losses,
        }
        return avg_loss, loader_iters, round_info

    def _gossip_average(self):
        """Gossip averaging among active workers."""
        all_vecs = torch.stack([params_to_vector(m) for m in self.models])
        new_vecs = self.W @ all_vecs
        for i in range(self.n_workers):
            if self.active[i]:
                vector_to_params(new_vecs[i], self.models[i])

    def get_average_model(self, active_only: bool = True) -> nn.Module:
        """Return average model across (active) workers."""
        if active_only:
            active_indices = [i for i in range(self.n_workers) if self.active[i]]
        else:
            active_indices = list(range(self.n_workers))
        if len(active_indices) == 0:
            return copy.deepcopy(self.models[0])

        vecs = torch.stack([params_to_vector(self.models[i]) for i in active_indices])
        avg_vec = vecs.mean(dim=0)
        avg_model = copy.deepcopy(self.models[0])
        vector_to_params(avg_vec, avg_model)
        return avg_model
