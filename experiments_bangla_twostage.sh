#!/bin/bash

# Model: google/mt5-base
# Dataset: R-RAD-Bengali
# Method: Two-Stage Reasoning (Bengali version)

# Set paths
DATA_DIR="../Medthink-Dataset/R-RAD-Bengali"
IMG_FEATURES="../Medthink-Dataset/R-RAD/detr.pth"
NAME_MAP="../Medthink-Dataset/R-RAD/name_map.json"

# Check if image features exist, extract if necessary
if [ ! -f "$IMG_FEATURES" ]; then
    echo "Image features not found. Extracting features first..."
    python extract_img_feature.py --dataset rad --image_dir ../Medthink-Dataset/R-RAD/images/ --output_dir ../Medthink-Dataset/R-RAD/
fi

# ==============================================================================
# ============================== Closed-End ====================================
# ==============================================================================
echo "=== Closed-End: First-Stage Training (Generating Rationales) ==="
CUDA_VISIBLE_DEVICES=0 python closed_end_train.py \
    --dataset rad \
    --method First-Stage_Reasoning \
    --epoch 150 \
    --lr 5e-4 \
    --bs 8 \
    --source_len 512 \
    --target_len 256 \
    --train_text_file_path $DATA_DIR/closed-end/trainset_bengali.json \
    --img_file_path $IMG_FEATURES \
    --img_name_map $NAME_MAP \
    --pretrained_model_path google/mt5-base \
    --output_dir bangla_closed_end_experiments \
    --rational

echo "=== Closed-End: First-Stage Generation (Creating Rationale Train/Test sets) ==="
CUDA_VISIBLE_DEVICES=0 python closed_end_generate.py \
    --dataset rad \
    --method First-Stage_Reasoning \
    --bs 8 \
    --source_len 512 \
    --target_len 256 \
    --text_file_path $DATA_DIR/closed-end/trainset_bengali.json \
    --model_path bangla_closed_end_experiments/First-Stage_Reasoning \
    --img_file_path $IMG_FEATURES \
    --img_name_map $NAME_MAP \
    --output_dir bangla_closed_end_experiments

CUDA_VISIBLE_DEVICES=0 python closed_end_generate.py \
    --dataset rad \
    --method First-Stage_Reasoning \
    --bs 8 \
    --source_len 512 \
    --target_len 256 \
    --text_file_path $DATA_DIR/closed-end/testset_bengali.json \
    --model_path bangla_closed_end_experiments/First-Stage_Reasoning \
    --img_file_path $IMG_FEATURES \
    --img_name_map $NAME_MAP \
    --output_dir bangla_closed_end_experiments

echo "=== Closed-End: Second-Stage Training (Generating Answers) ==="
CUDA_VISIBLE_DEVICES=0 python closed_end_train.py \
    --dataset rad \
    --method Second-Stage_Reasoning \
    --epoch 20 \
    --lr 5e-5 \
    --bs 8 \
    --source_len 512 \
    --target_len 32 \
    --train_text_file_path bangla_closed_end_experiments/First-Stage_Reasoning/train.json \
    --img_file_path $IMG_FEATURES \
    --img_name_map $NAME_MAP \
    --pretrained_model_path google/mt5-base \
    --output_dir bangla_closed_end_experiments

echo "=== Closed-End: Second-Stage Generation ==="
CUDA_VISIBLE_DEVICES=0 python closed_end_generate.py \
    --dataset rad \
    --method Second-Stage_Reasoning \
    --bs 8 \
    --source_len 512 \
    --target_len 32 \
    --text_file_path bangla_closed_end_experiments/First-Stage_Reasoning/test.json \
    --model_path bangla_closed_end_experiments/Second-Stage_Reasoning \
    --img_file_path $IMG_FEATURES \
    --img_name_map $NAME_MAP \
    --output_dir bangla_closed_end_experiments


# ==============================================================================
# ============================== Open-End ======================================
# ==============================================================================
echo "=== Open-End: First-Stage Training (Generating Rationales) ==="
CUDA_VISIBLE_DEVICES=0 python open_end_train.py \
    --dataset rad \
    --method First-Stage_Reasoning \
    --epoch 150 \
    --lr 5e-4 \
    --bs 8 \
    --source_len 512 \
    --target_len 256 \
    --train_text_file_path $DATA_DIR/open-end/trainset_bengali.json \
    --img_file_path $IMG_FEATURES \
    --img_name_map $NAME_MAP \
    --pretrained_model_path google/mt5-base \
    --output_dir bangla_open_end_experiments \
    --rational

echo "=== Open-End: First-Stage Generation (Creating Rationale Train/Test sets) ==="
CUDA_VISIBLE_DEVICES=0 python open_end_generate.py \
    --dataset rad \
    --method First-Stage_Reasoning \
    --bs 8 \
    --source_len 512 \
    --target_len 256 \
    --text_file_path $DATA_DIR/open-end/trainset_bengali.json \
    --model_path bangla_open_end_experiments/First-Stage_Reasoning \
    --img_file_path $IMG_FEATURES \
    --img_name_map $NAME_MAP \
    --output_dir bangla_open_end_experiments

CUDA_VISIBLE_DEVICES=0 python open_end_generate.py \
    --dataset rad \
    --method First-Stage_Reasoning \
    --bs 8 \
    --source_len 512 \
    --target_len 256 \
    --text_file_path $DATA_DIR/open-end/testset_bengali.json \
    --model_path bangla_open_end_experiments/First-Stage_Reasoning \
    --img_file_path $IMG_FEATURES \
    --img_name_map $NAME_MAP \
    --output_dir bangla_open_end_experiments

echo "=== Open-End: Second-Stage Training (Generating Answers) ==="
CUDA_VISIBLE_DEVICES=0 python open_end_train.py \
    --dataset rad \
    --method Second-Stage_Reasoning \
    --epoch 20 \
    --lr 5e-5 \
    --bs 8 \
    --source_len 512 \
    --target_len 32 \
    --train_text_file_path bangla_open_end_experiments/First-Stage_Reasoning/train.json \
    --img_file_path $IMG_FEATURES \
    --img_name_map $NAME_MAP \
    --pretrained_model_path google/mt5-base \
    --output_dir bangla_open_end_experiments

echo "=== Open-End: Second-Stage Generation ==="
CUDA_VISIBLE_DEVICES=0 python open_end_generate.py \
    --dataset rad \
    --method Second-Stage_Reasoning \
    --bs 8 \
    --source_len 512 \
    --target_len 32 \
    --text_file_path bangla_open_end_experiments/First-Stage_Reasoning/test.json \
    --model_path bangla_open_end_experiments/Second-Stage_Reasoning \
    --img_file_path $IMG_FEATURES \
    --img_name_map $NAME_MAP \
    --output_dir bangla_open_end_experiments
