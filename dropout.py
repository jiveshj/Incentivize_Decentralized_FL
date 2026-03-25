"""
Client dropout simulation.

Implements the paper's dropout model:
- After a warmup period (10% of training), clients evaluate whether
  the collaborative model meets their needs
- If F_i(x) > ρ_i, client i drops out
- Dropping clients severs their edges, potentially fragmenting the network
- Remaining connected components continue training independently
- Solo clients train locally
"""

import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import copy
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional

from topologies import metropolis_hastings_weights


def compute_client_losses(models: List[nn.Module],
                          loaders: List[DataLoader],
                          criterion: nn.Module,
                          active: List[bool],
                          device: str = "cpu") -> List[float]:
    """Compute empirical loss F_i(x_i) for each active client."""
    losses = [float('inf')] * len(models)
    for i, (model, loader) in enumerate(zip(models, loaders)):
        if not active[i] or loader is None:
            continue
        model.eval()
        total_loss, total_n = 0.0, 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                total_loss += criterion(output, target).item() * data.size(0)
                total_n += data.size(0)
        losses[i] = total_loss / max(total_n, 1)
    return losses

    
    """Compute accuracy for each active client for the dropout function."""
def compute_client_accuracies(
        models: List[nn.Module],
        loaders: List[DataLoader],
        active: List[bool],
        device: str = "cpu",
    ) -> List[float]:
    accuracies = [0.0] * len(models)
    for i, (model, loader) in enumerate(zip(models, loaders)):
        if not active[i] or loader is None:
            continue
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)                 
                preds = output.argmax(dim=1)        
                correct += (preds == target).sum().item()
                total += target.size(0)
        accuracies[i] = correct / max(total, 1)
    return accuracies


def determine_dropouts(client_losses_or_accuracies: List[float],
                       rho: List[float],
                       active: List[bool]) -> List[int]:
    """
    Determine which clients want to drop out.

    Client i drops out if F_i(x_i) > ρ_i (loss exceeds threshold).

    Returns list of client indices that want to drop out.
    """
    dropouts = []
    for i in range(len(client_losses_or_accuracies)):
        #print(f"Client_n {i}: loss_n={client_losses_or_accuracies[i]:.4f},"
        #                     f"ρ_n={rho[i]:.4f}")  
        #if active[i] and loss < threshold:
        if active[i] and client_losses_or_accuracies[i] < rho[i]:
            dropouts.append(i)
    return dropouts


def apply_dropouts(G: nx.Graph,
                   W: torch.Tensor,
                   dropout_nodes: List[int],
                   active: List[bool],
                   device: str = "cpu") -> Tuple[nx.Graph, torch.Tensor, List[bool], dict]:
    """
    Remove dropout nodes from the graph and recompute mixing matrix.

    When a node drops out:
    1. All its edges are severed
    2. The remaining graph may fragment into components
    3. Each connected component gets its own Metropolis-Hastings mixing matrix 
       and continues training collaboratively within that component
    4. Nodes not in any connected component become isolated and train solo (self-loop with weight 1)

    Returns:
        (new_graph, new_W, new_active, dropout_info)
    """
    N = W.shape[0]
    nodes = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Create subgraph without dropout nodes
    remaining_nodes = [n for n in nodes if node_to_idx[n] not in dropout_nodes
                       and active[node_to_idx[n]]]
    H = G.subgraph(remaining_nodes).copy()

    new_active = active.copy()
    for idx in dropout_nodes:
        new_active[idx] = False

    # Find connected components
    if len(remaining_nodes) == 0:
        components = []
    else:
        components = list(nx.connected_components(H))

    # Find largest connected component
    if len(components) > 0:
        largest_cc = max(components, key=len)
    else:
        largest_cc = set()

    # # Nodes not in largest CC become forcibly disconnected
    # forcibly_disconnected = []
    # for comp in components:
    #     if comp != largest_cc:
    #         for n in comp:
    #             idx = node_to_idx[n]
    #             new_active[idx] = False
    #             forcibly_disconnected.append(idx)


    # Build new mixing matrix for the active topology
    new_W = torch.zeros(N, N, device=device)
    # if len(largest_cc) > 1:
    #     sub_G = H.subgraph(largest_cc).copy()
    #     sub_W = metropolis_hastings_weights(sub_G)
    #     sub_nodes = sorted(sub_G.nodes())
    #     sub_node_to_idx = {n: i for i, n in enumerate(sub_nodes)}

    #     for n_i in sub_nodes:
    #         for n_j in sub_nodes:
    #             idx_i = node_to_idx[n_i]
    #             idx_j = node_to_idx[n_j]
    #             sub_i = sub_node_to_idx[n_i]
    #             sub_j = sub_node_to_idx[n_j]
    #             new_W[idx_i, idx_j] = sub_W[sub_i, sub_j]
    # elif len(largest_cc) == 1:
    #     # Single node: self-loop with weight 1
    #     n = list(largest_cc)[0]
    #     idx = node_to_idx[n]
    #     new_W[idx, idx] = 1.0

    # # Solo nodes get self-loop weight 1
    # for i in range(N):
    #     if not new_active[i]:
    #         new_W[i, :] = 0
    #         new_W[:, i] = 0
    #         new_W[i, i] = 1.0

    # new_G = H.subgraph(largest_cc).copy() if len(largest_cc) > 0 else nx.Graph()
    for comp in components:
        comp_nodes = sorted(comp)
        if len(comp_nodes) > 1:
            # Multi-node component: build MH mixing matrix for this subgraph
            sub_G = H.subgraph(comp_nodes).copy()
            sub_W = metropolis_hastings_weights(sub_G)
            sub_node_to_idx = {n: i for i, n in enumerate(comp_nodes)}

            for n_i in comp_nodes:
                for n_j in comp_nodes:
                    idx_i = node_to_idx[n_i]
                    idx_j = node_to_idx[n_j]
                    sub_i = sub_node_to_idx[n_i]
                    sub_j = sub_node_to_idx[n_j]
                    new_W[idx_i, idx_j] = sub_W[sub_i, sub_j]
        else:
            # Single isolated node: self-loop
            n = comp_nodes[0]
            idx = node_to_idx[n]
            new_W[idx, idx] = 1.0

    # Dropped-out nodes get self-loop weight 1 (solo training)
    for idx in dropout_nodes:
        new_W[idx, :] = 0
        new_W[:, idx] = 0
        new_W[idx, idx] = 1.0

    component_sizes = sorted([len(c) for c in components], reverse=True)

    dropout_info = {
        "voluntary_dropouts": dropout_nodes,
        "n_components_after": len(components),
        "component_sizes": component_sizes,
        "largest_cc_size": len(largest_cc),
        "total_active": sum(new_active),
        "retention_rate": sum(new_active) / N,
        # Retention in largest CC (paper's metric)
        "largest_cc_retention_rate": len(largest_cc) / N,
    }

    return H, new_W, new_active, dropout_info


    # dropout_info = {
    #     "voluntary_dropouts": dropout_nodes,
    #     "forcibly_disconnected": forcibly_disconnected,
    #     "n_components_after": len(components),
    #     "largest_cc_size": len(largest_cc),
    #     "total_active": sum(new_active),
    #     "retention_rate": sum(new_active) / N,
    # }

    # return new_G, new_W, new_active, dropout_info


def estimate_rho_local_training_accuracy(
    models: List[nn.Module],
    model_fn,
    train_loaders: List[DataLoader],
    val_loaders: List[DataLoader],
    criterion: nn.Module,
    n_workers: int,
    solo_rounds: int = 5,
    tau: int = 4,
    solo_lr: float = 0.01,
    device: str = "cpu",
) -> List[float]:
    """
    Estimate dropout thresholds ρ_i via local-only training, using accuracy.

    ρ_i = accuracy of a model trained only on client i's data,
    evaluated on its held-out validation set.
    """
    rho = []
    for i in range(n_workers):
        # Fresh solo model
        solo_model = model_fn().to(device)
        solo_model.train()
        loader_iter = iter(train_loaders[i])

        # Solo training
        for _ in range(solo_rounds * tau):
            try:
                data, target = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loaders[i])
                data, target = next(loader_iter)

            data, target = data.to(device), target.to(device)
            output = solo_model(data)
            loss = criterion(output, target)
            solo_model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in solo_model.parameters():
                    p.data -= solo_lr * p.grad

        # Evaluate accuracy on held-out validation data to get ρ_i
        solo_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in val_loaders[i]:
                data, target = data.to(device), target.to(device)
                output = solo_model(data)              # shape [B, C]
                preds = output.argmax(dim=1)          # predicted class
                correct += (preds == target).sum().item()
                total += target.size(0)

        acc = correct / max(total, 1)                 # in [0, 1]
        rho.append(acc)
        del solo_model

    return rho

def estimate_rho_local_training_loss(models: List[nn.Module],
                                model_fn, 
                                train_loaders: List[DataLoader],
                                val_loaders: List[DataLoader],
                                criterion: nn.Module,
                                n_workers: int,
                                solo_rounds: int,
                                tau: int,
                                solo_lr: float,
                                device: str = "cpu") -> List[float]:
    """
    Estimate dropout thresholds ρ_i via local-only training.

    Each client:
    1. Trains a FRESH model from scratch on their local TRAINING data
       (simulating what they'd achieve if they never joined collaboration)
    2. Evaluates the loss on held-out VALIDATION data
    This validation loss is ρ_i — the client's solo baseline.

    We use fresh initialization (not the collaborative model) because
    ρ_i should represent the counterfactual: "what if I trained alone
    from the start?" Starting from collaborative params would give an
    unfair head start, making ρ_i artificially low.

    Args:
        models: current client models (unused, kept for API consistency)
        model_fn: callable that returns a fresh nn.Module
        train_loaders: training data loaders per client (for solo training)
        val_loaders: held-out validation loaders per client (for evaluation)
        criterion: loss function
        n_workers: number of clients
        solo_rounds: number of rounds of solo training
        tau: local steps per round
        solo_lr: learning rate for solo training
        device: cpu/cuda


    Returns:
        list of ρ_i values
    """
    rho = []
    for i in range(n_workers):
        # Clone current model
        solo_model = model_fn().to(device)
        solo_model.train()
        loader_iter = iter(train_loaders[i])

        # Solo training
        for _ in range(solo_rounds * tau):
            try:
                data, target = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loaders[i])
                data, target = next(loader_iter)

            data, target = data.to(device), target.to(device)
            output = solo_model(data)
            loss = criterion(output, target)
            solo_model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in solo_model.parameters():
                    p.data -= solo_lr * p.grad

        # Evaluate on held-out validation data to get ρ_i
        solo_model.eval()
        total_loss, total_n = 0.0, 0
        with torch.no_grad():
            for data, target in val_loaders[i]:
                data, target = data.to(device), target.to(device)
                output = solo_model(data)
                total_loss += criterion(output, target).item() * data.size(0)
                total_n += data.size(0)

        rho.append(total_loss / max(total_n, 1))
        del solo_model

    return rho


def train_solo_models(models: List[nn.Module],
                      loaders: List[DataLoader],
                      dropped_out: List[int],
                      criterion: nn.Module,
                      n_rounds: int,
                      tau: int,
                      lr: float,
                      device: str = "cpu") -> List[nn.Module]:
    """
    Continue training for dropped-out clients on their local data only.

    Args:
        models: the models (will be modified in-place for dropped clients)
        loaders: per-client data loaders
        dropped_out: indices of dropped-out clients
        criterion: loss function
        n_rounds: number of solo training rounds
        tau: local steps per round
        lr: learning rate
        device: cpu/cuda

    Returns:
        Updated models list
    """
    for i in dropped_out:
        models[i].train()
        loader_iter = iter(loaders[i])

        for _ in range(n_rounds * tau):
            try:
                data, target = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loaders[i])
                data, target = next(loader_iter)

            data, target = data.to(device), target.to(device)
            output = models[i](data)
            loss = criterion(output, target)
            models[i].zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in models[i].parameters():
                    p.data -= lr * p.grad

    return models
