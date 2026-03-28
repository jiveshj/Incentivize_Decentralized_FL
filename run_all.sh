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
T=300
EVAL_EVERY=10

if [[ "$1" == "--quick" ]]; then
    QUICK=true
    T=30
    EVAL_EVERY=5
    echo ">>> QUICK MODE: T=$T, eval_every=$EVAL_EVERY"
fi

#TOPOLOGIES=(SFL18" "KCN" "SFL12" "3Con16")
#DATASETS=( "cifar10" "emnist" "fashionmnist") #test with celeba and cifar100 if there is time (first cifar100)
#SEEDS_CIFAR=(39 111 252)

TOPOLOGIES=("$1")
DATASETS=("$2")
SEEDS_CIFAR=("$3")
DEVICE="cpu"  # Change to "cuda" if GPU available
# Grids matching the paper
BS_LIST=(32) # test with 64 if there is time
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
#changing in femnist and emnist: lr=0.005 alg=1 and r=300 knc 32 fashionmnist and emnist
#121 - emnist
#149 - fashionmnist
#145 - cifar10
for TOPO in "${TOPOLOGIES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
      SEEDS=("${SEEDS_CIFAR[@]}")
        for SEED in "${SEEDS[@]}"; do
            #if [ "$DATASET" == "celeba" || "$DATASET" == "cifar10" ]; then
            #celeba
            LR=0.5
            #else
            #    LR=0.005
            #fi
            for BS in "${BS_LIST[@]}"; do
                for TAU in "${TAU_LIST[@]}"; do
                    echo ""
                    echo ""
                    echo ">>> [$TOPO / $DATASET] Running baseline only..."
                   
                    
                    echo ">>> [$TOPO / $DATASET] Running baseline with dropout..."
                    python3 run_experiment.py \
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
                        python3 run_experiment.py \
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

echo ""
echo "============================================"
echo "All experiments complete! Results in ./results/"
echo "============================================"

