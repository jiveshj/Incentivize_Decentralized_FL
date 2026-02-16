"""
Graph topologies and mixing matrix construction.

Implements the 4 topologies from the paper plus common baselines.
All mixing matrices are built using Metropolis-Hastings weights
to ensure symmetric, doubly-stochastic matrices.
"""

import torch
import numpy as np
import networkx as nx
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Metropolis-Hastings mixing matrix (used by the paper)
# ---------------------------------------------------------------------------

def metropolis_hastings_weights(G: nx.Graph) -> torch.Tensor:
    """
    Construct Metropolis-Hastings doubly-stochastic mixing matrix.

    w_{ij} = 1 / (1 + max(deg(i), deg(j)))   for i != j, (i,j) in E
    w_{ii} = 1 - sum_{j in N_i} w_{ij}

    This is the construction used in the paper (Section 5).
    """
    N = G.number_of_nodes()
    nodes = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    W = torch.zeros(N, N)

    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        deg_u = G.degree(u)
        deg_v = G.degree(v)
        weight = 1.0 / (1.0 + max(deg_u, deg_v))
        W[i, j] = weight
        W[j, i] = weight

    for i in range(N):
        W[i, i] = 1.0 - W[i].sum()

    return W


# ---------------------------------------------------------------------------
# Paper topologies
# ---------------------------------------------------------------------------

def scale_free_like_12() -> Tuple[nx.Graph, torch.Tensor]:
    """
    Scale Free Like 12 (SFL12) - 12 nodes with power-law-like degree
    distribution. Nodes 3 and 6 are critical to connectivity (bridge nodes).

    Based on Figure 2(a) of the paper and the description that nodes 3 and 6
    are connectivity-critical with high betweenness.
    """
    G = nx.Graph()
    G.add_nodes_from(range(12))

    # Hub nodes: 3 and 6 serve as bridges between clusters
    # Cluster 1: nodes 0, 1, 2 connected to hub 3
    # Cluster 2: nodes 4, 5 connected between hubs 3 and 6
    # Cluster 3: nodes 7, 8, 9, 10, 11 connected to hub 6
    edges = [
        # Cluster around node 3
        (0, 3), (1, 3), (2, 3), (0, 1), (1, 2),
        # Bridge between 3 and 6
        (3, 4), (4, 5), (5, 6), (3, 6),
        # Cluster around node 6
        (6, 7), (6, 8), (6, 9), (7, 8),
        (9, 10), (10, 11), (9, 11),
    ]
    G.add_edges_from(edges)
    W = metropolis_hastings_weights(G)
    return G, W


def three_connected_16() -> Tuple[nx.Graph, torch.Tensor]:
    """
    16 Node 3 Connections (3Con16) - 16 nodes, each with exactly degree 3.
    Regular structure means equal importance when using degree-based weights.
    More robust to disconnection.
    """
    # Use a 3-regular graph on 16 nodes
    # Constructing a specific 3-regular graph
    G = nx.Graph()
    G.add_nodes_from(range(16))

    # Create a 3-regular graph: ring + skip connections
    # Ring connections
    for i in range(16):
        G.add_edge(i, (i + 1) % 16)

    # Additional edges to make it 3-regular (each node needs one more edge)
    # Connect opposite-ish nodes
    skip_edges = [
        (0, 8), (1, 9), (2, 10), (3, 11),
        (4, 12), (5, 13), (6, 14), (7, 15),
    ]
    G.add_edges_from(skip_edges)

    # Verify 3-regular
    assert all(G.degree(n) == 3 for n in G.nodes()), \
        f"Not 3-regular: {dict(G.degree())}"

    W = metropolis_hastings_weights(G)
    return G, W


def scale_free_like_18() -> Tuple[nx.Graph, torch.Tensor]:
    """
    Scale Free Like 18 (SFL18) - 18 nodes, extension of SFL12 with
    power-law-inspired degree distribution.
    """
    G = nx.Graph()
    G.add_nodes_from(range(18))

    edges = [
        # Cluster 1: nodes 0, 1, 2 connected to hub 3
        (0, 3), (1, 3), (2, 3), (0, 1), (1, 2),
        # Bridge nodes
        (3, 4), (4, 5), (5, 6), (3, 6),
        # Central cluster around node 6
        (6, 7), (6, 8), (7, 8),
        # Extension: second bridge
        (8, 9), (9, 10),
        # Cluster 3: nodes 10-14
        (10, 11), (10, 12), (11, 12), (10, 13), (13, 14), (11, 14),
        # Extension cluster: nodes 15-17 connected to node 9
        (9, 15), (15, 16), (16, 17), (15, 17), (9, 16),
    ]
    G.add_edges_from(edges)
    W = metropolis_hastings_weights(G)
    return G, W


def karate_club_network() -> Tuple[nx.Graph, torch.Tensor]:
    """
    Zachary's Karate Club Network (KCN) - 34 nodes.
    Well-known social network with skewed degree distribution
    (avg degree 4.59, max 17, min 1).
    """
    G = nx.karate_club_graph()
    # Relabel nodes to 0..33 (already the case for networkx)
    W = metropolis_hastings_weights(G)
    return G, W


# ---------------------------------------------------------------------------
# Standard topologies (for additional experiments)
# ---------------------------------------------------------------------------

def ring_topology(n: int) -> Tuple[nx.Graph, torch.Tensor]:
    """Ring graph with n nodes."""
    G = nx.cycle_graph(n)
    W = metropolis_hastings_weights(G)
    return G, W


def fully_connected_topology(n: int) -> Tuple[nx.Graph, torch.Tensor]:
    """Fully connected graph (simulates federated averaging)."""
    G = nx.complete_graph(n)
    W = metropolis_hastings_weights(G)
    return G, W


def grid_topology(n: int) -> Tuple[nx.Graph, torch.Tensor]:
    """2D grid graph. n must be a perfect square."""
    side = int(np.sqrt(n))
    assert side * side == n, "n must be a perfect square for grid topology"
    G = nx.grid_2d_graph(side, side)
    # Relabel from (row,col) tuples to integers
    mapping = {(r, c): r * side + c for r in range(side) for c in range(side)}
    G = nx.relabel_nodes(G, mapping)
    W = metropolis_hastings_weights(G)
    return G, W


# ---------------------------------------------------------------------------
# Topology registry
# ---------------------------------------------------------------------------

def get_topology(name: str, n_workers: Optional[int] = None) -> Tuple[nx.Graph, torch.Tensor, int]:
    """
    Get a graph topology and its mixing matrix.

    Returns:
        (graph, mixing_matrix, n_workers)
    """
    # Paper-specific topologies (fixed sizes)
    fixed_topologies = {
        "SFL12": (scale_free_like_12, 12),
        "3Con16": (three_connected_16, 16),
        "SFL18": (scale_free_like_18, 18),
        "KCN": (karate_club_network, 34),
    }

    # Parameterized topologies
    param_topologies = {
        "ring": ring_topology,
        "fully_connected": fully_connected_topology,
        "grid": grid_topology,
    }

    if name in fixed_topologies:
        fn, expected_n = fixed_topologies[name]
        G, W = fn()
        if n_workers is not None and n_workers != expected_n:
            print(f"Warning: {name} topology has fixed size {expected_n}, "
                  f"ignoring n_workers={n_workers}")
        return G, W, expected_n

    elif name in param_topologies:
        assert n_workers is not None, f"n_workers required for '{name}' topology"
        G, W = param_topologies[name](n_workers)
        return G, W, n_workers

    else:
        raise ValueError(f"Unknown topology '{name}'. "
                         f"Available: {list(fixed_topologies.keys()) + list(param_topologies.keys())}")


# ---------------------------------------------------------------------------
# Topology analysis utilities
# ---------------------------------------------------------------------------

def analyze_topology(G: nx.Graph, W: torch.Tensor) -> dict:
    """Print and return key properties of a graph topology."""
    N = G.number_of_nodes()
    degrees = dict(G.degree())
    avg_deg = sum(degrees.values()) / N

    # Spectral gap: 1 - second largest eigenvalue of W
    eigenvalues = torch.linalg.eigvalsh(W)
    sorted_eigs = eigenvalues.sort(descending=True).values
    spectral_gap = 1 - sorted_eigs[1].item()

    # Centrality measures
    betweenness = nx.betweenness_centrality(G)
    eigenvector_cent = nx.eigenvector_centrality_numpy(G)

    # Connectivity
    node_connectivity = nx.node_connectivity(G)
    bridges = list(nx.bridges(G)) if not nx.is_directed(G) else []
    articulation_pts = list(nx.articulation_points(G))

    info = {
        "n_nodes": N,
        "n_edges": G.number_of_edges(),
        "avg_degree": avg_deg,
        "max_degree": max(degrees.values()),
        "min_degree": min(degrees.values()),
        "degrees": degrees,
        "spectral_gap": spectral_gap,
        "node_connectivity": node_connectivity,
        "n_bridges": len(bridges),
        "n_articulation_points": len(articulation_pts),
        "articulation_points": articulation_pts,
        "betweenness_centrality": betweenness,
        "eigenvector_centrality": eigenvector_cent,
    }
    return info


def print_topology_info(name: str, G: nx.Graph, W: torch.Tensor):
    """Pretty-print topology analysis."""
    info = analyze_topology(G, W)
    print(f"\n{'='*60}")
    print(f"Topology: {name}")
    print(f"{'='*60}")
    print(f"  Nodes            : {info['n_nodes']}")
    print(f"  Edges            : {info['n_edges']}")
    print(f"  Avg degree       : {info['avg_degree']:.2f}")
    print(f"  Max / Min degree : {info['max_degree']} / {info['min_degree']}")
    print(f"  Spectral gap (p) : {info['spectral_gap']:.4f}")
    print(f"  Node connectivity: {info['node_connectivity']}")
    print(f"  Articulation pts : {info['n_articulation_points']} "
          f"{info['articulation_points']}")
    print(f"  Bridges          : {info['n_bridges']}")

    # Doubly stochastic check
    row_err = (W.sum(dim=1) - 1.0).abs().max().item()
    col_err = (W.sum(dim=0) - 1.0).abs().max().item()
    sym_err = (W - W.T).abs().max().item()
    print(f"  W doubly-stoch   : row_err={row_err:.1e}, col_err={col_err:.1e}")
    print(f"  W symmetric      : err={sym_err:.1e}")
    print(f"{'='*60}")
    return info