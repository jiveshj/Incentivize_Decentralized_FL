"""
Weight strategies for importance weights a_i in NodeDrop-IDSGD.

This is the core module for the research question:
"What's the best strategy for assigning importance weights a_i?"

Strategies fall into three categories:
1. Topology-based: depend only on graph structure
2. Data-based: depend only on data heterogeneity
3. Hybrid: combine both topology and data information

All strategies implement the same interface:
    compute_weights(G, n_workers, client_distributions=None, **kwargs) -> np.ndarray
"""

import numpy as np
import networkx as nx
from typing import List, Optional, Dict, Callable
from data_utils import compute_kl_divergence, compute_label_entropy


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: Dict[str, Callable] = {}


def register_strategy(name: str):
    """Decorator to register a weight strategy."""
    def decorator(fn):
        _STRATEGY_REGISTRY[name] = fn
        return fn
    return decorator


def get_weight_strategy(name: str) -> Callable:
    """Get a registered weight strategy by name."""
    if name not in _STRATEGY_REGISTRY:
        raise ValueError(f"Unknown weight strategy '{name}'. "
                         f"Available: {list(_STRATEGY_REGISTRY.keys())}")
    return _STRATEGY_REGISTRY[name]


def list_strategies() -> List[str]:
    """List all registered weight strategies."""
    return list(_STRATEGY_REGISTRY.keys())


# ===========================================================================
# TOPOLOGY-BASED STRATEGIES
# ===========================================================================

@register_strategy("uniform")
def uniform_weights(G: nx.Graph, n_workers: int, **kwargs) -> np.ndarray:
    """
    Uniform weights: a_i = 1 for all nodes.
    Baseline — equivalent to [Cho et al. 2024] with equal importance.
    """
    return np.ones(n_workers)


@register_strategy("degree")
def degree_weights(G: nx.Graph, n_workers: int, **kwargs) -> np.ndarray:
    """
    Degree centrality: a_i = deg(i).
    The paper's default practical choice.
    Intuition: high-degree nodes are more important for connectivity.
    """
    nodes = sorted(G.nodes())
    return np.array([G.degree(n) for n in nodes], dtype=float)


@register_strategy("degree_normalized")
def degree_normalized_weights(G: nx.Graph, n_workers: int, **kwargs) -> np.ndarray:
    """
    Normalized degree: a_i = deg(i) / avg_degree.
    Scales so average weight is 1.
    """
    nodes = sorted(G.nodes())
    degrees = np.array([G.degree(n) for n in nodes], dtype=float)
    avg = degrees.mean()
    return degrees / avg if avg > 0 else degrees


@register_strategy("betweenness")
def betweenness_weights(G: nx.Graph, n_workers: int, **kwargs) -> np.ndarray:
    """
    Betweenness centrality: a_i = betweenness(i) * N + 1.
    Captures "bridge" nodes whose removal fragments the network.
    +1 ensures no node has zero weight.
    """
    nodes = sorted(G.nodes())
    bc = nx.betweenness_centrality(G)
    weights = np.array([bc[n] * n_workers + 1.0 for n in nodes])
    return weights


@register_strategy("eigenvector")
def eigenvector_weights(G: nx.Graph, n_workers: int, **kwargs) -> np.ndarray:
    """
    Eigenvector centrality: a_i proportional to eigenvector centrality.
    Important if connected to other important nodes.
    Scaled so mean = 1.
    """
    nodes = sorted(G.nodes())
    try:
        ec = nx.eigenvector_centrality_numpy(G)
    except nx.NetworkXError:
        # Fallback for disconnected graphs
        ec = {n: 1.0 / n_workers for n in nodes}
    weights = np.array([ec[n] for n in nodes])
    weights = weights / weights.mean()  # normalize so mean=1
    return weights


@register_strategy("closeness")
def closeness_weights(G: nx.Graph, n_workers: int, **kwargs) -> np.ndarray:
    """
    Closeness centrality: a_i = closeness(i) * N.
    Nodes that are "close" to all others are important for fast information
    propagation in gossip.
    """
    nodes = sorted(G.nodes())
    cc = nx.closeness_centrality(G)
    weights = np.array([cc[n] * n_workers for n in nodes])
    return weights


@register_strategy("pagerank")
def pagerank_weights(G: nx.Graph, n_workers: int, **kwargs) -> np.ndarray:
    """
    PageRank: a_i = pagerank(i) * N.
    Random-walk-based importance measure.
    """
    nodes = sorted(G.nodes())
    pr = nx.pagerank(G)
    weights = np.array([pr[n] * n_workers for n in nodes])
    return weights


@register_strategy("vertex_connectivity")
def vertex_connectivity_weights(G: nx.Graph, n_workers: int, **kwargs) -> np.ndarray:
    """
    Vertex connectivity contribution: measures how much removing node i
    reduces the graph's vertex connectivity.
    a_i = 1 + (original_connectivity - connectivity_without_i)

    Directly measures impact on network robustness.
    Expensive for large graphs.
    """
    nodes = sorted(G.nodes())
    original_kappa = nx.node_connectivity(G)
    weights = np.ones(n_workers)

    for idx, n in enumerate(nodes):
        H = G.copy()
        H.remove_node(n)
        if H.number_of_nodes() > 0 and nx.is_connected(H):
            reduced_kappa = nx.node_connectivity(H)
        else:
            reduced_kappa = 0
        # +1 baseline so even non-critical nodes participate
        weights[idx] = 1.0 + max(0, original_kappa - reduced_kappa)

    return weights


@register_strategy("articulation")
def articulation_point_weights(G: nx.Graph, n_workers: int,
                                ap_weight: float = 3.0, **kwargs) -> np.ndarray:
    """
    Articulation point indicator: a_i = ap_weight if node i is an
    articulation point, else 1.

    Simple binary: critical nodes get boosted weight.
    """
    nodes = sorted(G.nodes())
    aps = set(nx.articulation_points(G))
    weights = np.array([ap_weight if n in aps else 1.0 for n in nodes])
    return weights


# ===========================================================================
# DATA-BASED STRATEGIES
# ===========================================================================

@register_strategy("kl_divergence")
def kl_divergence_weights(G: nx.Graph, n_workers: int,
                           client_distributions: List[np.ndarray] = None,
                           **kwargs) -> np.ndarray:
    """
    KL divergence from global distribution: a_i = 1 + KL(p_i || p_global).
    Clients whose data differs most from global are highest flight risks,
    so we incentivize them more.
    """
    assert client_distributions is not None, "Need client_distributions"
    dists = np.array(client_distributions)
    global_dist = dists.mean(axis=0)
    kl_divs = np.array([compute_kl_divergence(d, global_dist) for d in dists])
    return 1.0 + kl_divs


@register_strategy("inverse_entropy")
def inverse_entropy_weights(G: nx.Graph, n_workers: int,
                             client_distributions: List[np.ndarray] = None,
                             **kwargs) -> np.ndarray:
    """
    Inverse label entropy: a_i = 1 + (max_entropy - entropy_i) / max_entropy.
    Low-entropy clients have very skewed data and are flight risks.
    """
    assert client_distributions is not None, "Need client_distributions"
    n_classes = len(client_distributions[0])
    max_entropy = np.log(n_classes)
    entropies = np.array([compute_label_entropy(d) for d in client_distributions])
    return 1.0 + (max_entropy - entropies) / max_entropy


@register_strategy("dropout_risk")
def dropout_risk_weights(G: nx.Graph, n_workers: int,
                          client_distributions: List[np.ndarray] = None,
                          **kwargs) -> np.ndarray:
    """
    Estimated dropout risk based on data heterogeneity.
    Combines KL divergence (how different) with entropy (how specialized).
    a_i = 1 + KL_i * (1 - entropy_i/max_entropy)
    """
    assert client_distributions is not None, "Need client_distributions"
    dists = np.array(client_distributions)
    global_dist = dists.mean(axis=0)
    n_classes = len(dists[0])
    max_entropy = np.log(n_classes)

    kl_divs = np.array([compute_kl_divergence(d, global_dist) for d in dists])
    entropies = np.array([compute_label_entropy(d) for d in dists])
    specialization = 1.0 - entropies / max_entropy  # 0 = uniform, 1 = single class

    return 1.0 + kl_divs * specialization


# ===========================================================================
# HYBRID STRATEGIES (topology + data)
# ===========================================================================

@register_strategy("degree_x_kl")
def degree_x_kl_weights(G: nx.Graph, n_workers: int,
                          client_distributions: List[np.ndarray] = None,
                          **kwargs) -> np.ndarray:
    """
    Degree × KL divergence: a_i = deg(i) * (1 + KL_i).
    A node is critical if both structurally important AND has heterogeneous data.
    """
    deg_w = degree_weights(G, n_workers)
    kl_w = kl_divergence_weights(G, n_workers, client_distributions)
    return deg_w * kl_w


@register_strategy("betweenness_x_kl")
def betweenness_x_kl_weights(G: nx.Graph, n_workers: int,
                               client_distributions: List[np.ndarray] = None,
                               **kwargs) -> np.ndarray:
    """
    Betweenness × KL: a_i = (1 + betweenness_i * N) * (1 + KL_i).
    Targets bridge nodes with heterogeneous data.
    """
    btw_w = betweenness_weights(G, n_workers)
    kl_w = kl_divergence_weights(G, n_workers, client_distributions)
    return btw_w * kl_w


@register_strategy("betweenness_x_dropout_risk")
def betweenness_x_dropout_risk_weights(G: nx.Graph, n_workers: int,
                                        client_distributions: List[np.ndarray] = None,
                                        **kwargs) -> np.ndarray:
    """
    Betweenness × dropout risk.
    """
    btw_w = betweenness_weights(G, n_workers)
    dr_w = dropout_risk_weights(G, n_workers, client_distributions)
    return btw_w * dr_w


@register_strategy("eigenvector_x_kl")
def eigenvector_x_kl_weights(G: nx.Graph, n_workers: int,
                               client_distributions: List[np.ndarray] = None,
                               **kwargs) -> np.ndarray:
    """
    Eigenvector centrality × KL divergence.
    """
    eig_w = eigenvector_weights(G, n_workers)
    kl_w = kl_divergence_weights(G, n_workers, client_distributions)
    return eig_w * kl_w


@register_strategy("pagerank_x_kl")
def pagerank_x_kl_weights(G: nx.Graph, n_workers: int,
                            client_distributions: List[np.ndarray] = None,
                            **kwargs) -> np.ndarray:
    """
    PageRank × KL divergence.
    """
    pr_w = pagerank_weights(G, n_workers)
    kl_w = kl_divergence_weights(G, n_workers, client_distributions)
    return pr_w * kl_w


@register_strategy("articulation_x_kl")
def articulation_x_kl_weights(G: nx.Graph, n_workers: int,
                                client_distributions: List[np.ndarray] = None,
                                **kwargs) -> np.ndarray:
    """
    Articulation point boost × KL divergence.
    """
    ap_w = articulation_point_weights(G, n_workers)
    kl_w = kl_divergence_weights(G, n_workers, client_distributions)
    return ap_w * kl_w


# ---------------------------------------------------------------------------
# Helper to summarize weight strategies
# ---------------------------------------------------------------------------

def compute_and_print_weights(strategy_name: str, G: nx.Graph, n_workers: int,
                               client_distributions: List[np.ndarray] = None):
    """Compute and display weights for a given strategy."""
    fn = get_weight_strategy(strategy_name)
    weights = fn(G, n_workers, client_distributions=client_distributions)

    print(f"\n  Strategy: {strategy_name}")
    print(f"  Weights : {np.round(weights, 3)}")
    print(f"  Mean={weights.mean():.3f}, Std={weights.std():.3f}, "
          f"Max={weights.max():.3f}, Min={weights.min():.3f}")
    return weights