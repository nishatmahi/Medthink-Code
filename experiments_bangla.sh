#!/bin/bash

# Model: google/mt5-base
# Dataset: R-RAD-Bengali

# Set paths
DATA_DIR="../Medthink-Dataset/R-RAD-Bengali"
IMG_FEATURES="../Medthink-Dataset/R-RAD/detr.pth"
NAME_MAP="../Medthink-Dataset/R-RAD/name_map.json"

# Note: You must run extract_img_feature.py first to generate detr.pth and name_map.json if they don't exist.
# Example: python extract_img_feature.py --dataset rad --image_dir ../Medthink-Dataset/R-RAD/images/ --output_dir ../Medthink-Dataset/R-RAD/

# ==============================================================================
# ============================== Closed-End ====================================
# ==============================================================================

# Method: Explanation
CUDA_VISIBLE_DEVICES=0 python closed_end_train.py --dataset rad --method Explanation --epoch 150 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path $DATA_DIR/closed-end/trainset_bengali.json --img_file_path $IMG_FEATURES --img_name_map $NAME_MAP --pretrained_model_path google/mt5-base --output_dir bangla_closed_end_experiments
CUDA_VISIBLE_DEVICES=0 python closed_end_generate.py --dataset rad --method Explanation --bs 8 --source_len 512 --target_len 256 --text_file_path $DATA_DIR/closed-end/testset_bengali.json --model_path bangla_closed_end_experiments/Explanation --img_file_path $IMG_FEATURES --img_name_map $NAME_MAP --output_dir bangla_closed_end_experiments

# Method: Reasoning
CUDA_VISIBLE_DEVICES=0 python closed_end_train.py --dataset rad --method Reasoning --epoch 150 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path $DATA_DIR/closed-end/trainset_bengali.json --img_file_path $IMG_FEATURES --img_name_map $NAME_MAP --pretrained_model_path google/mt5-base --output_dir bangla_closed_end_experiments
CUDA_VISIBLE_DEVICES=0 python closed_end_generate.py --dataset rad --method Reasoning --bs 8 --source_len 512 --target_len 256 --text_file_path $DATA_DIR/closed-end/testset_bengali.json --model_path bangla_closed_end_experiments/Reasoning --img_file_path $IMG_FEATURES --img_name_map $NAME_MAP --output_dir bangla_closed_end_experiments

# ==============================================================================
# ============================== Open-End ======================================
# ==============================================================================

# Method: Explanation
CUDA_VISIBLE_DEVICES=0 python open_end_train.py --dataset rad --method Explanation --epoch 150 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path $DATA_DIR/open-end/trainset_bengali.json --img_file_path $IMG_FEATURES --img_name_map $NAME_MAP --pretrained_model_path google/mt5-base --output_dir bangla_open_end_experiments
CUDA_VISIBLE_DEVICES=0 python open_end_generate.py --dataset rad --method Explanation --bs 8 --source_len 512 --target_len 256 --text_file_path $DATA_DIR/open-end/testset_bengali.json --model_path bangla_open_end_experiments/Explanation --img_file_path $IMG_FEATURES --img_name_map $NAME_MAP --output_dir bangla_open_end_experiments

# Method: Reasoning
CUDA_VISIBLE_DEVICES=0 python open_end_train.py --dataset rad --method Reasoning --epoch 150 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path $DATA_DIR/open-end/trainset_bengali.json --img_file_path $IMG_FEATURES --img_name_map $NAME_MAP --pretrained_model_path google/mt5-base --output_dir bangla_open_end_experiments
CUDA_VISIBLE_DEVICES=0 python open_end_generate.py --dataset rad --method Reasoning --bs 8 --source_len 512 --target_len 256 --text_file_path $DATA_DIR/open-end/testset_bengali.json --model_path bangla_open_end_experiments/Reasoning --img_file_path $IMG_FEATURES --img_name_map $NAME_MAP --output_dir bangla_open_end_experiments
