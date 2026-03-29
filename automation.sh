#!/bin/bash

# This file contains ALL combinations:
# screen sh run_all.sh TOPOLOGY DATASET SEED
# ---------- CIFAR100 ----------
#screen -dmS SFL18_cifar100_0 bash run_all.sh SFL18 cifar100 0
#screen -dmS SFL18_cifar100_1 bash run_all.sh SFL18 cifar100 1
#screen -dmS SFL18_cifar100_2 bash run_all.sh SFL18 cifar100 2

#screen -dmS KCN_cifar100_30 bash run_all.sh KCN cifar100 30
#screen -dmS KCN_cifar100_31 bash run_all.sh KCN cifar100 31
#screen -dmS KCN_cifar100_32 bash run_all.sh KCN cifar100 32

#screen -dmS SFL12_cifar100_0 bash run_all.sh SFL12 cifar100 0
#screen -dmS SFL12_cifar100_1 bash run_all.sh SFL12 cifar100 1
#screen -dmS SFL12_cifar100_2 bash run_all.sh SFL12 cifar100 2

#screen -dmS 3Con16_cifar100_0 bash run_all.sh 3Con16 cifar100 0
#screen -dmS 3Con16_cifar100_1 bash run_all.sh 3Con16 cifar100 1
#screen -dmS 3Con16_cifar100_2 bash run_all.sh 3Con16 cifar100 2

# ---------- CELEBA ----------
bash run_all.sh SFL12 celeba 100
bash run_all.sh SFL12 celeba 101
bash run_all.sh SFL12 celeba 102
bash run_all.sh SFL12 celeba 103

bash run_all.sh SFL18 celeba 104
bash run_all.sh SFL18 celeba 105
bash run_all.sh SFL18 celeba 106
bash run_all.sh SFL18 celeba 107

bash run_all.sh KCN celeba 108
bash run_all.sh KCN celeba 109
bash run_all.sh KCN celeba 110
bash run_all.sh KCN celeba 111

bash run_all.sh 3Con16 celeba 112
bash run_all.sh 3Con16 celeba 113
bash run_all.sh 3Con16 celeba 114
bash run_all.sh 3Con16 celeba 115

# ---------- EMNIST 121 ----------

#screen -dmS SFL18_emnist_0     bash run_all.sh SFL18 emnist 0
#screen -dmS SFL18_emnist_1     bash run_all.sh SFL18 emnist 1
#screen -dmS SFL18_emnist_2     bash run_all.sh SFL18 emnist 2

#screen -dmS KCN_emnist_0       bash run_all.sh KCN emnist 0
#screen -dmS KCN_emnist_1       bash run_all.sh KCN emnist 1
#screen -dmS KCN_emnist_2       bash run_all.sh KCN emnist 2

#screen -dmS SFL12_emnist_0     bash run_all.sh SFL12 emnist 0
#screen -dmS SFL12_emnist_1     bash run_all.sh SFL12 emnist 1
#screen -dmS SFL12_emnist_2     bash run_all.sh SFL12 emnist 2

#screen -dmS 3Con16_emnist_0    bash run_all.sh 3Con16 emnist 0
#screen -dmS 3Con16_emnist_1    bash run_all.sh 3Con16 emnist 1
#screen -dmS 3Con16_emnist_2    bash run_all.sh 3Con16 emnist 2

# ---------- FASHIONMNIST 149 ----------

#screen -dmS SFL18_fashion_0    bash run_all.sh SFL18 fashionmnist 0
#screen -dmS SFL18_fashion_1    bash run_all.sh SFL18 fashionmnist 1
#screen -dmS SFL18_fashion_2    bash run_all.sh SFL18 fashionmnist 2

#screen -dmS KCN_fashion_0      bash run_all.sh KCN fashionmnist 0
#screen -dmS KCN_fashion_1      bash run_all.sh KCN fashionmnist 1
#screen -dmS KCN_fashion_2      bash run_all.sh KCN fashionmnist 2

#screen -dmS SFL12_fashion_0    bash run_all.sh SFL12 fashionmnist 0
#screen -dmS SFL12_fashion_1    bash run_all.sh SFL12 fashionmnist 1
#screen -dmS SFL12_fashion_2    bash run_all.sh SFL12 fashionmnist 2

#screen -dmS 3Con16_fashion_0   bash run_all.sh 3Con16 fashionmnist 0
#screen -dmS 3Con16_fashion_1   bash run_all.sh 3Con16 fashionmnist 1
#screen -dmS 3Con16_fashion_2   bash run_all.sh 3Con16 fashionmnist 2
