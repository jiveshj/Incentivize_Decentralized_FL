#!/bin/bash
# =================================================================
# Run the full experiment grid matching the paper's setup.
#
# Topologies: SFL12, 3Con16, SFL18, KCN
# Datasets:   fashionmnist, emnist, cifar10, cifar100, celeba, mnist
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

TOPOLOGIES=("KCN" "3Con16" "SFL18" "SFL12")
DATASETS=("fashionmnist" "emnist" "cifar10") #test with celeba and cifar100 if there is time (first cifar100)
SEEDS_CIFAR=(60 61 62)

DEVICE="cpu"  # Change to "cuda" if GPU available

# Grids matching the paper
LR_LIST=(0.05) # test with 0.01 and 0.1 if there is time
BS_LIST=(64) # test with 32 if there is time
TAU_LIST=(5) # test with 10 if there is time
GAMMA_LIST=(0.9524)  # β in {5,10,20} test with 0.8333 0.9091 for 5 and 10 if there is time !!!important
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
      SEEDS=("${SEEDS_CIFAR[@]}")
        for SEED in "${SEEDS[@]}"; do
            for LR in "${LR_LIST[@]}"; do
                for BS in "${BS_LIST[@]}"; do
                    for TAU in "${TAU_LIST[@]}"; do
                        echo ""
                        echo ">>> [$TOPO / $DATASET] Running baseline only..."
                        python run_experiment.py \
                            --algorithm baseline \
                            --topology "$TOPO" \
                            --dataset "$DATASET" \
                            --lr "$LR" \
                            --seed "$SEED" \
                            --batch_size "$BS" \
                            --tau "$TAU" \
                            --T "$T" \
                            --eval_every "$EVAL_EVERY" \
                            --device "$DEVICE" \
                            --output_dir results
                        echo ""
                        echo ">>> [$TOPO / $DATASET] Running baseline with dropout..."
                        python run_experiment.py \
                            --algorithm baseline_dropout \
                            --topology "$TOPO" \
                            --dataset "$DATASET" \
                            --lr "$LR" \
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

#o que tem os 4 testes é o 30662                     
#tenho que deixar correr para emnist e cifar10 com kcn   