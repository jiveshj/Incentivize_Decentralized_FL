import os, json, glob, csv
import numpy as np

RESULTS_DIR = "results"
OUT_CSV = os.path.join(RESULTS_DIR, "summary.csv")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    # group by (dataset, topology, algorithm, lr, batch_size, tau, gamma, weight_strategy)
    groups = {}
    
    for path in glob.glob(os.path.join(RESULTS_DIR, "**", "*.json"), recursive=True):
        data = load_json(path)
        args = data.get("args")
        history = data.get("history")
        strategies = data.get("strategies")

        # Skip comparison files (weight_strategy=ALL summary from nodedrop)
        if strategies is not None:
            continue

        if args is None or history is None:
            continue

        key = (
            args["dataset"],
            args["topology"],
            args["algorithm"],
            float(args["lr"]),
            int(args["batch_size"]),
            int(args["tau"]),
            float(args.get("gamma", 0.0)) if "gamma" in args else None,
            args.get("weight_strategy", None),
        )

        groups.setdefault(key, []).append((int(args["seed"]), history))

    # write CSV
    fieldnames = [
        "dataset", "topology", "algorithm",
        "lr", "batch_size", "tau",
        "gamma", "weight_strategy",
        "num_seeds", "expected_seeds", "acc_seed1", "acc_seed2", "acc_seed3",
        "avg_preferred_acc_mean", "avg_preferred_acc_std",
        "final_retention_rate_mean", "final_retention_rate_std",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for key, runs in sorted(groups.items()):
            dataset, topo, algo, lr, bs, tau, gamma, weight_strategy = key

            expected_seeds = 3

            histories = [h for (_, h) in runs]

            # metrics from history (fallbacks if not present)
            avg_pref_list = [h.get("avg_preferred_acc") for h in histories
                             if "avg_preferred_acc" in h]
            if avg_pref_list:
                mean_pref = float(np.mean(avg_pref_list))
                std_pref = float(np.std(avg_pref_list))
            else:
                mean_pref = std_pref = float("nan")

            final_ret_list = [h.get("final_retention_rate") for h in histories
                              if "final_retention_rate" in h]
            if final_ret_list:
                mean_ret = float(np.mean(final_ret_list))
                std_ret = float(np.std(final_ret_list))
            else:
                mean_ret = std_ret = float("nan")

            row = {
                "dataset": dataset,
                "topology": topo,
                "algorithm": algo,
                "lr": lr,
                "batch_size": bs,
                "tau": tau,
                "gamma": gamma if gamma is not None else "",
                "weight_strategy": weight_strategy if weight_strategy is not None else "",
                "num_seeds": len(runs),
                "expected_seeds": expected_seeds,
                "acc_seed1": avg_pref_list[0],
                "acc_seed2": avg_pref_list[1] if len(avg_pref_list) > 1 else None,
                "acc_seed3": avg_pref_list[2] if len(avg_pref_list) > 2 else None,
                "avg_preferred_acc_mean": mean_pref,
                "avg_preferred_acc_std": std_pref,
                "final_retention_rate_mean": mean_ret,
                "final_retention_rate_std": std_ret,
            }
            writer.writerow(row)

if __name__ == "__main__":
    main()
