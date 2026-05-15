#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

DATA_ROOT=~/data/fastgs/datasets/mipnerf360

FLAGS="--eval --densification_interval 500 --adaptive_densification --structural_weight 5.0 --densify_until_iter 18000"

echo "========================================================"
echo "STARTING ADAPTIVE EDGE-MASKING BENCHMARK"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================================"

mkdir -p output

echo "[1/9] Bicycle (Adaptive)..."
python train.py -s $DATA_ROOT/bicycle $FLAGS --grad_abs_thresh 0.0008 --model_path output/bicycle_Adaptive
python render.py -m output/bicycle_Adaptive --skip_train --quiet
python metrics.py -m output/bicycle_Adaptive

echo "[2/9] Kitchen (Adaptive)..."
python train.py -s $DATA_ROOT/kitchen $FLAGS --highfeature_lr 0.02 --grad_abs_thresh 0.0002 --model_path output/kitchen_Adaptive
python render.py -m output/kitchen_Adaptive --skip_train --quiet
python metrics.py -m output/kitchen_Adaptive

echo "[3/9] Room (Adaptive)..."
python train.py -s $DATA_ROOT/room $FLAGS --highfeature_lr 0.02 --grad_abs_thresh 0.0004 --model_path output/room_Adaptive
python render.py -m output/room_Adaptive --skip_train --quiet
python metrics.py -m output/room_Adaptive

echo "[4/9] Garden (Adaptive)..."
python train.py -s $DATA_ROOT/garden $FLAGS --highfeature_lr 0.02 --loss_thresh 0.06 --grad_abs_thresh 0.0003 --model_path output/garden_Adaptive
python render.py -m output/garden_Adaptive --skip_train --quiet
python metrics.py -m output/garden_Adaptive

echo "[5/9] Bonsai (Adaptive)..."
python train.py -s $DATA_ROOT/bonsai $FLAGS --highfeature_lr 0.02 --grad_abs_thresh 0.0002 --model_path output/bonsai_Adaptive
python render.py -m output/bonsai_Adaptive --skip_train --quiet
python metrics.py -m output/bonsai_Adaptive

echo "[6/9] Counter (Adaptive)..."
python train.py -s $DATA_ROOT/counter $FLAGS --highfeature_lr 0.02 --grad_abs_thresh 0.0004 --model_path output/counter_Adaptive
python render.py -m output/counter_Adaptive --skip_train --quiet
python metrics.py -m output/counter_Adaptive

echo "[7/9] Stump (Adaptive)..."
python train.py -s $DATA_ROOT/stump $FLAGS --dense 0.004 --grad_abs_thresh 0.001 --model_path output/stump_Adaptive
python render.py -m output/stump_Adaptive --skip_train --quiet
python metrics.py -m output/stump_Adaptive

echo "[8/9] Treehill (Adaptive)..."
python train.py -s $DATA_ROOT/treehill $FLAGS --dense 0.01 --grad_abs_thresh 0.0018 --model_path output/treehill_Adaptive
python render.py -m output/treehill_Adaptive --skip_train --quiet
python metrics.py -m output/treehill_Adaptive

echo "[9/9] Flowers (Adaptive)..."
python train.py -s $DATA_ROOT/flowers $FLAGS --dense 0.005 --grad_abs_thresh 0.001 --model_path output/flowers_Adaptive
python render.py -m output/flowers_Adaptive --skip_train --quiet
python metrics.py -m output/flowers_Adaptive

echo "BENCHMARK COMPLETE."
