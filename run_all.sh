#!/bin/bash
# =================================================================
# Run the full experiment grid matching the paper's setup.
#
# Topologies: SFL12, 3Con16, SFL18, KCN
# Datasets:   fashionmnist, emnist, cifar10
# Algorithms: baseline_dropout, nodedrop (with all weight strategies)
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh              # Full grid
#   ./run_all.sh --quick      # Quick test with fewer rounds
# =================================================================

set -e

QUICK=false
T=200
EVAL_EVERY=10

if [[ "$1" == "--quick" ]]; then
    QUICK=true
    T=30
    EVAL_EVERY=5
    echo ">>> QUICK MODE: T=$T, eval_every=$EVAL_EVERY"
fi

TOPOLOGIES=("SFL12" "3Con16" "SFL18" "KCN")
DATASETS=("fashionmnist" "emnist" "cifar10")
DEVICE="cpu"  # Change to "cuda" if GPU available

# Ensure output directory exists
mkdir -p results

echo "============================================"
echo "Running Full Experiment Grid"
echo "Topologies: ${TOPOLOGIES[*]}"
echo "Datasets:   ${DATASETS[*]}"
echo "T=$T, Device=$DEVICE"
echo "============================================"

for TOPO in "${TOPOLOGIES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo ">>> [$TOPO / $DATASET] Running baseline with dropout..."
        python run_experiment.py \
            --algorithm baseline_dropout \
            --topology "$TOPO" \
            --dataset "$DATASET" \
            --T "$T" \
            --eval_every "$EVAL_EVERY" \
            --device "$DEVICE" \
            --output_dir results

        echo ""
        echo ">>> [$TOPO / $DATASET] Running NodeDrop-IDSGD with ALL weight strategies..."
        python run_experiment.py \
            --algorithm nodedrop \
            --weight_strategy ALL \
            --topology "$TOPO" \
            --dataset "$DATASET" \
            --T "$T" \
            --eval_every "$EVAL_EVERY" \
            --device "$DEVICE" \
            --output_dir results
    done
done

echo ""
echo "============================================"
echo "All experiments complete! Results in ./results/"
echo "============================================"