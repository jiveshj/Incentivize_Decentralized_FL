#!/bin/bash

# This file contains ALL combinations:
# screen sh run_all.sh TOPOLOGY DATASET SEED

# ---------- CIFAR10 ----------
screen -dmS SFL18_cifar10_0    bash run_all.sh SFL18 cifar10 0
screen -dmS SFL18_cifar10_1    bash run_all.sh SFL18 cifar10 1
screen -dmS SFL18_cifar10_2    bash run_all.sh SFL18 cifar10 2

screen -dmS KCN_cifar10_0      bash run_all.sh KCN cifar10 0
screen -dmS KCN_cifar10_1      bash run_all.sh KCN cifar10 1
screen -dmS KCN_cifar10_2      bash run_all.sh KCN cifar10 2


screen -dmS SFL12_cifar10_0    bash run_all.sh SFL12 cifar10 0
screen -dmS SFL12_cifar10_1    bash run_all.sh SFL12 cifar10 1
screen -dmS SFL12_cifar10_2    bash run_all.sh SFL12 cifar10 2

screen -dmS 3Con16_cifar10_0   bash run_all.sh 3Con16 cifar10 0
screen -dmS 3Con16_cifar10_1   bash run_all.sh 3Con16 cifar10 1
screen -dmS 3Con16_cifar10_2   bash run_all.sh 3Con16 cifar10 2

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
