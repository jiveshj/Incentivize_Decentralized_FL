"""
Main experiment runner for Incentivized Decentralized SGD.

Usage:
    # Baseline (Algorithm 1) - no dropout
    python run_experiment.py --algorithm baseline --dataset fashionmnist --topology SFL12

    # Baseline with dropout simulation
    python run_experiment.py --algorithm baseline_dropout --dataset fashionmnist --topology SFL12

    # NodeDrop-IDSGD (Algorithm 2) with degree weights
    python run_experiment.py --algorithm nodedrop --weight_strategy degree --topology SFL12

    # Compare all weight strategies on a topology
    python run_experiment.py --algorithm nodedrop --weight_strategy ALL --topology SFL12

    # Quick smoke test
    python run_experiment.py --algorithm baseline --dataset mnist --topology ring --n_workers 4 --T 20
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.data import DataLoader, Subset

from utils import (set_seed, evaluate_model, evaluate_per_client,
                   consensus_distance, save_results, count_parameters,
                   generate_experiment_name)
from topologies import get_topology, print_topology_info, metropolis_hastings_weights
from data_utils import (get_dataset, get_alpha_for_dataset, partition_dirichlet,
                        create_client_test_loaders, analyze_data_heterogeneity)
from models import get_model_fn
from algorithms import DecentralizedSGD, NodeDropIDSGD
from weight_strategies import get_weight_strategy, list_strategies
from dropout import (estimate_rho_local_training, determine_dropouts,
                     apply_dropouts, compute_client_losses, train_solo_models)


def run_baseline(args, model_fn, train_loaders, test_loaders, test_dataset,
                 W, n_workers, G, client_distributions, device):
    """
    Run Algorithm 1: Decentralized SGD (no dropout).
    All clients participate for all T rounds.
    """
    criterion = nn.CrossEntropyLoss()
    global_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    dsgd = DecentralizedSGD(
        model_fn=model_fn,
        n_workers=n_workers,
        mixing_matrix=W,
        tau=args.tau,
        device=device,
    )

    history = {
        "train_loss": [], "global_test_loss": [], "global_test_acc": [],
        "per_client_acc": [], "consensus": [],
    }
    loader_iters = None

    for t in range(args.T):
        # Learning rate schedule
        if args.lr_schedule == "cosine":
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * t / args.T))
        elif args.lr_schedule == "step":
            lr = args.lr * (0.1 ** (t // (args.T // 3)))
        else:
            lr = args.lr

        train_loss, loader_iters = dsgd.train_round(
            train_loaders, lr, criterion, loader_iters
        )
        history["train_loss"].append(train_loss)

        if (t + 1) % args.eval_every == 0 or t == args.T - 1:
            # Global evaluation (average model)
            avg_model = dsgd.get_average_model()
            global_result = evaluate_model(avg_model, global_test_loader,
                                           criterion, device)

            # Per-client evaluation
            client_results = evaluate_per_client(
                dsgd.models, test_loaders, criterion, device
            )
            client_accs = [r["accuracy"] for r in client_results]
            cd = consensus_distance(dsgd.models)

            history["global_test_loss"].append(global_result["loss"])
            history["global_test_acc"].append(global_result["accuracy"])
            history["per_client_acc"].append(client_accs)
            history["consensus"].append(cd)

            if args.verbose:
                print(f"Round {t+1:>4d}/{args.T} | lr={lr:.5f} | "
                      f"train_loss={train_loss:.4f} | "
                      f"test_acc={global_result['accuracy']*100:.2f}% | "
                      f"avg_client_acc={np.mean(client_accs)*100:.2f}% | "
                      f"consensus={cd:.6f}")

    return history, dsgd


def run_with_dropout(args, model_fn, train_loaders, test_loaders, test_dataset,
                     W, n_workers, G, client_distributions, device,
                     algorithm_class, importance_weights=None):
    """
    Run with client dropout simulation.

    Works for both baseline (DecentralizedSGD) and NodeDrop-IDSGD:
    1. Train for warmup_rounds (10% of T)
    2. Estimate ρ_i and allow dropouts
    3. Continue training with modified topology
    4. Solo-train dropped clients
    5. Evaluate preferred model accuracy
    """
    criterion = nn.CrossEntropyLoss()
    global_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    warmup_rounds = int(args.T * args.dropout_warmup_frac)

    # Initialize algorithm
    if algorithm_class == NodeDropIDSGD:
        algo = NodeDropIDSGD(
            model_fn=model_fn,
            n_workers=n_workers,
            mixing_matrix=W.clone(),
            importance_weights=importance_weights,
            gamma=args.gamma,
            epsilon=args.epsilon,
            tau=args.tau,
            tau_eta=args.tau_eta,
            device=device,
        )
    else:
        algo = DecentralizedSGD(
            model_fn=model_fn,
            n_workers=n_workers,
            mixing_matrix=W.clone(),
            tau=args.tau,
            device=device,
        )

    history = {
        "train_loss": [], "global_test_acc": [], "per_client_acc": [],
        "consensus": [], "dropout_events": [], "retention_rate": [],
    }
    loader_iters = None
    all_dropped = []  # accumulate all dropped clients
    current_W = W.clone().to(device)
    current_G = G.copy()
    active = [True] * n_workers

    for t in range(args.T):
        # Learning rate schedule
        if args.lr_schedule == "cosine":
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * t / args.T))
        elif args.lr_schedule == "step":
            lr = args.lr * (0.1 ** (t // (args.T // 3)))
        else:
            lr = args.lr

        # Update mixing matrix in the algorithm
        if algorithm_class == NodeDropIDSGD:
            algo.W = current_W
            train_loss, loader_iters, round_info = algo.train_round(
                train_loaders, lr, criterion, loader_iters
            )
        else:
            algo.W = current_W
            train_loss, loader_iters = algo.train_round(
                train_loaders, lr, criterion, loader_iters
            )

        history["train_loss"].append(train_loss)

        # --- Dropout check after warmup ---
        if t == warmup_rounds and t > 0:
            # Estimate ρ_i
            rho = estimate_rho_local_training(
                algo.models, train_loaders, criterion, n_workers,
                solo_rounds=args.rho_solo_rounds, tau=args.tau,
                solo_lr=lr, device=device
            )

            # Compute current losses
            client_losses = compute_client_losses(
                algo.models, train_loaders, criterion, active, device
            )

            # Determine who wants to drop
            dropouts = determine_dropouts(client_losses, rho, active)

            if len(dropouts) > 0:
                if args.verbose:
                    print(f"\n>>> Round {t+1}: Clients {dropouts} want to drop out")
                    for d in dropouts:
                        print(f"    Client {d}: loss={client_losses[d]:.4f}, "
                              f"ρ={rho[d]:.4f}")

                # Apply dropouts
                current_G, current_W, active, dropout_info = apply_dropouts(
                    G, current_W, dropouts, active, device
                )

                # Update algorithm state
                algo.active = active

                if args.verbose:
                    print(f"    Voluntary dropouts: {dropout_info['voluntary_dropouts']}")
                    print(f"    Forcibly disconnected: {dropout_info['forcibly_disconnected']}")
                    print(f"    Largest CC size: {dropout_info['largest_cc_size']}/{n_workers}")
                    print(f"    Retention rate: {dropout_info['retention_rate']*100:.1f}%\n")

                all_dropped.extend(dropouts)
                all_dropped.extend(dropout_info["forcibly_disconnected"])
                history["dropout_events"].append({
                    "round": t + 1,
                    **dropout_info,
                })

        history["retention_rate"].append(sum(active) / n_workers)

        # --- Evaluation ---
        if (t + 1) % args.eval_every == 0 or t == args.T - 1:
            # Per-client evaluation on each client's test set
            client_results = evaluate_per_client(
                algo.models, test_loaders, criterion, device
            )
            client_accs = [r["accuracy"] for r in client_results]
            cd = consensus_distance([algo.models[i] for i in range(n_workers)
                                     if active[i]]) if sum(active) > 1 else 0.0

            history["per_client_acc"].append(client_accs)
            history["consensus"].append(cd)

            # Global test on average of active models
            avg_model = algo.get_average_model(active_only=True)
            global_result = evaluate_model(avg_model, global_test_loader,
                                           criterion, device)
            history["global_test_acc"].append(global_result["accuracy"])

            if args.verbose:
                active_accs = [client_accs[i] for i in range(n_workers) if active[i]]
                dropped_accs = [client_accs[i] for i in range(n_workers) if not active[i]]
                print(f"Round {t+1:>4d}/{args.T} | "
                      f"active={sum(active):>2d}/{n_workers} | "
                      f"avg_active_acc={np.mean(active_accs)*100:.2f}% | "
                      f"avg_all_acc={np.mean(client_accs)*100:.2f}%")

    # --- Post-training: solo train dropped clients ---
    if len(all_dropped) > 0 and args.solo_train_rounds > 0:
        if args.verbose:
            print(f"\n>>> Solo training for dropped clients: {list(set(all_dropped))}")
        remaining_rounds = args.T - warmup_rounds
        solo_models = train_solo_models(
            algo.models, train_loaders, list(set(all_dropped)), criterion,
            n_rounds=remaining_rounds, tau=args.tau, lr=args.lr * 0.5,
            device=device
        )

    # --- Final evaluation: preferred model accuracy ---
    final_client_results = evaluate_per_client(
        algo.models, test_loaders, criterion, device
    )
    preferred_accs = []
    for i in range(n_workers):
        preferred_accs.append(final_client_results[i]["accuracy"])

    history["preferred_model_acc"] = preferred_accs
    history["avg_preferred_acc"] = float(np.mean(preferred_accs))
    history["final_retention_rate"] = sum(active) / n_workers
    history["dropped_clients"] = list(set(all_dropped))
    history["active_clients"] = [i for i in range(n_workers) if active[i]]

    if args.verbose:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"  Avg preferred model accuracy: {np.mean(preferred_accs)*100:.2f}%")
        print(f"  Client retention rate:        {sum(active)/n_workers*100:.1f}%")
        print(f"  Active clients:               {[i for i in range(n_workers) if active[i]]}")
        print(f"  Dropped clients:              {list(set(all_dropped))}")
        print(f"{'='*60}")

    return history, algo


def main():
    parser = argparse.ArgumentParser(
        description="Incentivized Decentralized SGD Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Algorithm
    parser.add_argument("--algorithm", type=str, default="baseline",
                        choices=["baseline", "baseline_dropout", "nodedrop"],
                        help="Algorithm to run")
    parser.add_argument("--weight_strategy", type=str, default="degree",
                        help="Weight strategy for a_i (use ALL to compare all)")

    # Topology
    parser.add_argument("--topology", type=str, default="SFL12",
                        help="Graph topology")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Number of workers (auto-set for paper topologies)")

    # Data
    parser.add_argument("--dataset", type=str, default="fashionmnist",
                        choices=["fashionmnist", "emnist", "cifar10", "mnist"])
    parser.add_argument("--alpha", type=float, default=None,
                        help="Dirichlet alpha (None = use paper default)")
    parser.add_argument("--data_dir", type=str, default="./data")

    # Training
    parser.add_argument("--T", type=int, default=200, help="Communication rounds")
    parser.add_argument("--tau", type=int, default=4, help="Local SGD steps")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["none", "cosine", "step"])

    # NodeDrop-IDSGD specific
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Interpolation between ERM and dropout penalty")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Smoothing in LR denominator")
    parser.add_argument("--tau_eta", type=int, default=10,
                        help="Gossip steps for scaling factor estimation")

    # Dropout simulation
    parser.add_argument("--dropout_warmup_frac", type=float, default=0.1,
                        help="Fraction of T before dropout is allowed")
    parser.add_argument("--rho_solo_rounds", type=int, default=5,
                        help="Solo training rounds to estimate rho_i")
    parser.add_argument("--solo_train_rounds", type=int, default=1,
                        help="Solo training flag for dropped clients (0=disable)")

    # Evaluation
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--test_samples_per_client", type=int, default=500)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", default=False)

    args = parser.parse_args()
    if args.quiet:
        args.verbose = False

    set_seed(args.seed)
    device = args.device

    # --- Setup topology ---
    G, W, n_workers = get_topology(args.topology, args.n_workers)
    args.n_workers = n_workers
    topo_info = print_topology_info(args.topology, G, W)

    # --- Setup data ---
    alpha = args.alpha or get_alpha_for_dataset(args.dataset)
    train_dataset, test_dataset, num_classes = get_dataset(args.dataset, args.data_dir)
    model_fn = get_model_fn(args.dataset, num_classes)

    print(f"\nDataset: {args.dataset} | Classes: {num_classes} | "
          f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    print(f"Model params: {count_parameters(model_fn()):,}")
    print(f"Dirichlet alpha: {alpha}")

    # Partition data
    client_indices, client_distributions = partition_dirichlet(
        train_dataset, n_workers, alpha, seed=args.seed
    )
    hetero_info = analyze_data_heterogeneity(client_distributions)
    print(f"Data heterogeneity: avg_KL={hetero_info['avg_kl']:.3f}, "
          f"max_KL={hetero_info['max_kl']:.3f}")

    # Build loaders
    train_subsets = [Subset(train_dataset, idx) for idx in client_indices]
    train_loaders = [
        DataLoader(sub, batch_size=args.batch_size, shuffle=True, drop_last=False)
        for sub in train_subsets
    ]
    test_loaders = create_client_test_loaders(
        test_dataset, client_distributions, n_workers,
        test_samples_per_client=args.test_samples_per_client,
        seed=args.seed
    )

    # --- Run experiment ---
    print(f"\nAlgorithm: {args.algorithm}")
    print(f"T={args.T}, tau={args.tau}, lr={args.lr}, schedule={args.lr_schedule}")
    print(f"{'='*60}\n")

    if args.algorithm == "baseline":
        history, algo = run_baseline(
            args, model_fn, train_loaders, test_loaders, test_dataset,
            W, n_workers, G, client_distributions, device
        )

    elif args.algorithm == "baseline_dropout":
        history, algo = run_with_dropout(
            args, model_fn, train_loaders, test_loaders, test_dataset,
            W, n_workers, G, client_distributions, device,
            algorithm_class=DecentralizedSGD
        )

    elif args.algorithm == "nodedrop":
        if args.weight_strategy == "ALL":
            # Run all strategies and compare
            all_results = {}
            for strategy_name in list_strategies():
                print(f"\n{'#'*60}")
                print(f"# Weight Strategy: {strategy_name}")
                print(f"{'#'*60}")
                set_seed(args.seed)  # Reset seed for fair comparison

                strategy_fn = get_weight_strategy(strategy_name)
                weights = strategy_fn(G, n_workers,
                                      client_distributions=client_distributions)
                print(f"  Weights: {np.round(weights, 2)}")

                history, algo = run_with_dropout(
                    args, model_fn, train_loaders, test_loaders, test_dataset,
                    W, n_workers, G, client_distributions, device,
                    algorithm_class=NodeDropIDSGD,
                    importance_weights=weights,
                )
                all_results[strategy_name] = {
                    "avg_preferred_acc": history["avg_preferred_acc"],
                    "final_retention_rate": history["final_retention_rate"],
                    "preferred_accs": history["preferred_model_acc"],
                }

            # Summary comparison
            print(f"\n{'='*70}")
            print(f"STRATEGY COMPARISON SUMMARY")
            print(f"{'='*70}")
            print(f"{'Strategy':<30s} {'Avg Acc':>10s} {'Retention':>10s}")
            print(f"{'-'*50}")
            for name, res in sorted(all_results.items(),
                                     key=lambda x: -x[1]["avg_preferred_acc"]):
                print(f"{name:<30s} {res['avg_preferred_acc']*100:>9.2f}% "
                      f"{res['final_retention_rate']*100:>9.1f}%")

            # Save all
            save_path = os.path.join(args.output_dir,
                                     f"comparison_{args.dataset}_{args.topology}.json")
            save_results({"args": vars(args), "strategies": all_results}, save_path)
            return

        else:
            strategy_fn = get_weight_strategy(args.weight_strategy)
            weights = strategy_fn(G, n_workers,
                                  client_distributions=client_distributions)
            print(f"Weight strategy: {args.weight_strategy}")
            print(f"Weights: {np.round(weights, 3)}")

            history, algo = run_with_dropout(
                args, model_fn, train_loaders, test_loaders, test_dataset,
                W, n_workers, G, client_distributions, device,
                algorithm_class=NodeDropIDSGD,
                importance_weights=weights,
            )

    # --- Save results ---
    exp_name = generate_experiment_name(args)
    save_path = os.path.join(args.output_dir, f"{exp_name}.json")
    results = {
        "args": vars(args),
        "topology_info": {
            "n_nodes": topo_info["n_nodes"],
            "n_edges": topo_info["n_edges"],
            "spectral_gap": topo_info["spectral_gap"],
            "articulation_points": topo_info["articulation_points"],
        },
        "data_heterogeneity": hetero_info,
        "history": history,
    }
    save_results(results, save_path)


if __name__ == "__main__":
    main()