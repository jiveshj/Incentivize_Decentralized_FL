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

# Grids matching the paper
LR_LIST=(0.05 0.03 0.01 0.005 0.003 0.001)
BS_LIST=(32 64)
TAU_LIST=(5 10)
GAMMA_LIST=(0.8333 0.9091 0.9524 0.9804)  # β in {5,10,20,50}
SEEDS_CIFAR=(0 1 2)
SEEDS_OTHERS=(0 1 2 3 4 5)
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
    if [[ "$DATASET" == "cifar10" ]]; then
      SEEDS=("${SEEDS_CIFAR[@]}")
    else
      SEEDS=("${SEEDS_OTHERS[@]}")
    fi

        for SEED in "${SEEDS[@]}"; do
            for LR in "${LR_LIST[@]}"; do
                for BS in "${BS_LIST[@]}"; do
                    for TAU in "${TAU_LIST[@]}"; do
                        echo ""
                        echo ">>> [$TOPO / $DATASET] Running baseline with dropout..."
                        python run_experiment.py \
                            --algorithm baseline_dropout \
                            --topology "$TOPO" \
                            --dataset "$DATASET" \
                            --lr "$LR" \
                            --gamma "0" \
                            --seed "$SEED" \
                            --batch_size "$BS" \
                            --tau "$TAU" \
                            --T "$T" \
                            --eval_every "$EVAL_EVERY" \
                            --device "$DEVICE" \
                            --output_dir results

                        echo ""
                        echo ">>> [$TOPO / $DATASET] Running NodeDrop-IDSGD with ALL weight strategies..."
            
                        for GAMMA in "${GAMMA_LIST[@]}"; do
                            python run_experiment.py \
                            --algorithm nodedrop \
                            --weight_strategy ALL \
                            --topology "$TOPO" \
                            --dataset "$DATASET" \
                            --T "$T" \
                            --eval_every "$EVAL_EVERY" \
                            --lr "$LR" \
                            --seed "$SEED" \
                            --batch_size "$BS" \
                            --tau "$TAU" \
                            --gamma "$GAMMA" \
                            --tau_eta 5 \
                            --device "$DEVICE" \
                            --output_dir results
                        done
                    done
                done
            done
        done
    done
done

echo ""
echo "============================================"
echo "All experiments complete! Results in ./results/"
echo "============================================"