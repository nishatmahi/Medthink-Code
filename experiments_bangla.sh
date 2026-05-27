#!/bin/bash
# =============================================================================
# Two-Stage Reasoning Training Pipeline — All Bengali Datasets
# Datasets: R-RAD-Bengali | R-SLAKE-Bangla | R-PathVQA-Bengali
# =============================================================================
# IMPORTANT: Before running, ensure DETR image features are extracted for
# each dataset using extract_img_feature.py and set the paths below.
# =============================================================================
set -e  # exit on error

# ─── GPU Configuration ───────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0

# ─── Image Feature Paths (update output_dir after running extract_img_feature.py) ─
RAD_IMG_FEATURES="../Medthink-Dataset/R-RAD-Bengali/detr.pth"
RAD_NAME_MAP="../Medthink-Dataset/R-RAD-Bengali/name_map.json"

SLAKE_IMG_FEATURES="../Medthink-Dataset/R-SLAKE-Bangla/detr.pth"
SLAKE_NAME_MAP="../Medthink-Dataset/R-SLAKE-Bangla/name_map.json"

PATH_IMG_FEATURES="../Medthink-Dataset/R-PathVQA-Bengali/detr.pth"
PATH_NAME_MAP="../Medthink-Dataset/R-PathVQA-Bengali/name_map.json"

# ─── Data Directories ────────────────────────────────────────────────────────
RAD_CLOSED_DIR="../Medthink-Dataset/R-RAD-Bengali/closed-end"
RAD_OPEN_DIR="../Medthink-Dataset/R-RAD-Bengali/open-end"
RAD_CKPT_DIR="../Medthink-Dataset/R-RAD-Bengali/.checkpoints"

SLAKE_CLOSED_DIR="../Medthink-Dataset/R-SLAKE-Bangla/closed-end"
SLAKE_OPEN_DIR="../Medthink-Dataset/R-SLAKE-Bangla/open-end"
SLAKE_CKPT_DIR="../Medthink-Dataset/R-SLAKE-Bangla/.checkpoints"

PATH_CLOSED_DIR="../Medthink-Dataset/R-PathVQA-Bengali/closed-end"
PATH_OPEN_DIR="../Medthink-Dataset/R-PathVQA-Bengali/open-end"
PATH_CKPT_DIR="../Medthink-Dataset/R-PathVQA-Bengali/.checkpoints"

# ─── Base Model ──────────────────────────────────────────────────────────────
BASE_MODEL="google/flan-t5-base"

# ─── Training Hyperparameters ────────────────────────────────────────────────
EPOCH=10
BS=8
GRAD_ACCUM=2
LR=5e-5
WD=1e-2
SEED=42
SOURCE_LEN=512
TARGET_LEN=256
EVAL_BS=16


# =============================================================================
#  R-RAD-Bengali
# =============================================================================
echo "============================================================"
echo "  R-RAD-Bengali — Closed-End"
echo "============================================================"

# ── RAD | Closed | Explanation ────────────────────────────────────────────────
echo "[RAD-Closed] Step 1/5: Training Explanation..."
python closed_end_train.py \
  --train_text_file_path "$RAD_CLOSED_DIR/trainset_bengali.json" \
  --img_file_path        "$RAD_IMG_FEATURES" \
  --img_name_map         "$RAD_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$RAD_CKPT_DIR/closed-end" \
  --dataset              rad \
  --method               Explanation \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

# ── RAD | Closed | without_R ─────────────────────────────────────────────────
echo "[RAD-Closed] Step 2/5: Training without_R..."
python closed_end_train.py \
  --train_text_file_path "$RAD_CLOSED_DIR/trainset_bengali.json" \
  --img_file_path        "$RAD_IMG_FEATURES" \
  --img_name_map         "$RAD_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$RAD_CKPT_DIR/closed-end" \
  --dataset              rad \
  --method               without_R \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

# ── RAD | Closed | Reasoning ──────────────────────────────────────────────────
echo "[RAD-Closed] Step 3/5: Training Reasoning..."
python closed_end_train.py \
  --train_text_file_path "$RAD_CLOSED_DIR/trainset_bengali.json" \
  --img_file_path        "$RAD_IMG_FEATURES" \
  --img_name_map         "$RAD_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$RAD_CKPT_DIR/closed-end" \
  --dataset              rad \
  --method               Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

# ── RAD | Closed | Two-Stage: First-Stage Train ───────────────────────────────
echo "[RAD-Closed] Step 4/5: First-Stage Training (rationale generation)..."
python closed_end_train.py \
  --train_text_file_path "$RAD_CLOSED_DIR/trainset_bengali.json" \
  --img_file_path        "$RAD_IMG_FEATURES" \
  --img_name_map         "$RAD_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$RAD_CKPT_DIR/closed-end" \
  --dataset              rad \
  --method               First-Stage_Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM \
  --rational

# ── RAD | Closed | Generate Train Rationales ─────────────────────────────────
echo "[RAD-Closed] Injecting rationales into train set..."
mkdir -p "$RAD_CKPT_DIR/closed-end/First-Stage_Reasoning"
python closed_end_generate.py \
  --text_file_path "$RAD_CLOSED_DIR/trainset_bengali.json" \
  --img_file_path  "$RAD_IMG_FEATURES" \
  --img_name_map   "$RAD_NAME_MAP" \
  --model_path     "$RAD_CKPT_DIR/closed-end/First-Stage_Reasoning" \
  --output_dir     "$RAD_CKPT_DIR/closed-end" \
  --dataset        rad \
  --method         First-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

# ── RAD | Closed | Generate Test Rationales ───────────────────────────────────
echo "[RAD-Closed] Injecting rationales into test set..."
python closed_end_generate.py \
  --text_file_path "$RAD_CLOSED_DIR/testset_bengali.json" \
  --img_file_path  "$RAD_IMG_FEATURES" \
  --img_name_map   "$RAD_NAME_MAP" \
  --model_path     "$RAD_CKPT_DIR/closed-end/First-Stage_Reasoning" \
  --output_dir     "$RAD_CKPT_DIR/closed-end" \
  --dataset        rad \
  --method         First-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

# ── RAD | Closed | Two-Stage: Second-Stage Train ─────────────────────────────
echo "[RAD-Closed] Step 5/5: Second-Stage Training (final answer)..."
python closed_end_train.py \
  --train_text_file_path "$RAD_CKPT_DIR/closed-end/First-Stage_Reasoning/train.json" \
  --img_file_path        "$RAD_IMG_FEATURES" \
  --img_name_map         "$RAD_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$RAD_CKPT_DIR/closed-end" \
  --dataset              rad \
  --method               Second-Stage_Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

# ── RAD | Closed | Generate Final Predictions ────────────────────────────────
echo "[RAD-Closed] Generating final predictions..."
python closed_end_generate.py \
  --text_file_path "$RAD_CKPT_DIR/closed-end/First-Stage_Reasoning/test.json" \
  --img_file_path  "$RAD_IMG_FEATURES" \
  --img_name_map   "$RAD_NAME_MAP" \
  --model_path     "$RAD_CKPT_DIR/closed-end/Second-Stage_Reasoning" \
  --output_dir     "$RAD_CKPT_DIR/closed-end" \
  --dataset        rad \
  --method         Second-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

echo ""
echo "============================================================"
echo "  R-RAD-Bengali — Open-End"
echo "============================================================"

# ── RAD | Open | Explanation ─────────────────────────────────────────────────
echo "[RAD-Open] Step 1/5: Training Explanation..."
python open_end_train.py \
  --train_text_file_path "$RAD_OPEN_DIR/trainset_bengali.json" \
  --img_file_path        "$RAD_IMG_FEATURES" \
  --img_name_map         "$RAD_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$RAD_CKPT_DIR/open-end" \
  --dataset              rad \
  --method               Explanation \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

# ── RAD | Open | without_R ───────────────────────────────────────────────────
echo "[RAD-Open] Step 2/5: Training without_R..."
python open_end_train.py \
  --train_text_file_path "$RAD_OPEN_DIR/trainset_bengali.json" \
  --img_file_path        "$RAD_IMG_FEATURES" \
  --img_name_map         "$RAD_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$RAD_CKPT_DIR/open-end" \
  --dataset              rad \
  --method               without_R \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

# ── RAD | Open | Reasoning ───────────────────────────────────────────────────
echo "[RAD-Open] Step 3/5: Training Reasoning..."
python open_end_train.py \
  --train_text_file_path "$RAD_OPEN_DIR/trainset_bengali.json" \
  --img_file_path        "$RAD_IMG_FEATURES" \
  --img_name_map         "$RAD_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$RAD_CKPT_DIR/open-end" \
  --dataset              rad \
  --method               Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

# ── RAD | Open | First-Stage Train ──────────────────────────────────────────
echo "[RAD-Open] Step 4/5: First-Stage Training..."
python open_end_train.py \
  --train_text_file_path "$RAD_OPEN_DIR/trainset_bengali.json" \
  --img_file_path        "$RAD_IMG_FEATURES" \
  --img_name_map         "$RAD_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$RAD_CKPT_DIR/open-end" \
  --dataset              rad \
  --method               First-Stage_Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM \
  --rational

# ── RAD | Open | Generate Train Rationales ───────────────────────────────────
echo "[RAD-Open] Injecting rationales into train set..."
mkdir -p "$RAD_CKPT_DIR/open-end/First-Stage_Reasoning"
python open_end_generate.py \
  --text_file_path "$RAD_OPEN_DIR/trainset_bengali.json" \
  --img_file_path  "$RAD_IMG_FEATURES" \
  --img_name_map   "$RAD_NAME_MAP" \
  --model_path     "$RAD_CKPT_DIR/open-end/First-Stage_Reasoning" \
  --output_dir     "$RAD_CKPT_DIR/open-end" \
  --dataset        rad \
  --method         First-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

# ── RAD | Open | Generate Test Rationales ────────────────────────────────────
echo "[RAD-Open] Injecting rationales into test set..."
python open_end_generate.py \
  --text_file_path "$RAD_OPEN_DIR/testset_bengali.json" \
  --img_file_path  "$RAD_IMG_FEATURES" \
  --img_name_map   "$RAD_NAME_MAP" \
  --model_path     "$RAD_CKPT_DIR/open-end/First-Stage_Reasoning" \
  --output_dir     "$RAD_CKPT_DIR/open-end" \
  --dataset        rad \
  --method         First-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

# ── RAD | Open | Second-Stage Train ─────────────────────────────────────────
echo "[RAD-Open] Step 5/5: Second-Stage Training..."
python open_end_train.py \
  --train_text_file_path "$RAD_CKPT_DIR/open-end/First-Stage_Reasoning/train.json" \
  --img_file_path        "$RAD_IMG_FEATURES" \
  --img_name_map         "$RAD_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$RAD_CKPT_DIR/open-end" \
  --dataset              rad \
  --method               Second-Stage_Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

# ── RAD | Open | Generate Final Predictions ──────────────────────────────────
echo "[RAD-Open] Generating final predictions..."
python open_end_generate.py \
  --text_file_path "$RAD_CKPT_DIR/open-end/First-Stage_Reasoning/test.json" \
  --img_file_path  "$RAD_IMG_FEATURES" \
  --img_name_map   "$RAD_NAME_MAP" \
  --model_path     "$RAD_CKPT_DIR/open-end/Second-Stage_Reasoning" \
  --output_dir     "$RAD_CKPT_DIR/open-end" \
  --dataset        rad \
  --method         Second-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED


# =============================================================================
#  R-SLAKE-Bangla
# NOTE: SLAKE uses train_bengali.json / test_bengali.json (no "set" suffix)
# =============================================================================
echo ""
echo "============================================================"
echo "  R-SLAKE-Bangla — Closed-End"
echo "============================================================"

echo "[SLAKE-Closed] Step 1/5: Training Explanation..."
python closed_end_train.py \
  --train_text_file_path "$SLAKE_CLOSED_DIR/train_bengali.json" \
  --img_file_path        "$SLAKE_IMG_FEATURES" \
  --img_name_map         "$SLAKE_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$SLAKE_CKPT_DIR/closed-end" \
  --dataset              slake \
  --method               Explanation \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

echo "[SLAKE-Closed] Step 2/5: Training without_R..."
python closed_end_train.py \
  --train_text_file_path "$SLAKE_CLOSED_DIR/train_bengali.json" \
  --img_file_path        "$SLAKE_IMG_FEATURES" \
  --img_name_map         "$SLAKE_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$SLAKE_CKPT_DIR/closed-end" \
  --dataset              slake \
  --method               without_R \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

echo "[SLAKE-Closed] Step 3/5: Training Reasoning..."
python closed_end_train.py \
  --train_text_file_path "$SLAKE_CLOSED_DIR/train_bengali.json" \
  --img_file_path        "$SLAKE_IMG_FEATURES" \
  --img_name_map         "$SLAKE_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$SLAKE_CKPT_DIR/closed-end" \
  --dataset              slake \
  --method               Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

echo "[SLAKE-Closed] Step 4/5: First-Stage Training..."
python closed_end_train.py \
  --train_text_file_path "$SLAKE_CLOSED_DIR/train_bengali.json" \
  --img_file_path        "$SLAKE_IMG_FEATURES" \
  --img_name_map         "$SLAKE_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$SLAKE_CKPT_DIR/closed-end" \
  --dataset              slake \
  --method               First-Stage_Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM \
  --rational

mkdir -p "$SLAKE_CKPT_DIR/closed-end/First-Stage_Reasoning"
python closed_end_generate.py \
  --text_file_path "$SLAKE_CLOSED_DIR/train_bengali.json" \
  --img_file_path  "$SLAKE_IMG_FEATURES" \
  --img_name_map   "$SLAKE_NAME_MAP" \
  --model_path     "$SLAKE_CKPT_DIR/closed-end/First-Stage_Reasoning" \
  --output_dir     "$SLAKE_CKPT_DIR/closed-end" \
  --dataset        slake \
  --method         First-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

python closed_end_generate.py \
  --text_file_path "$SLAKE_CLOSED_DIR/test_bengali.json" \
  --img_file_path  "$SLAKE_IMG_FEATURES" \
  --img_name_map   "$SLAKE_NAME_MAP" \
  --model_path     "$SLAKE_CKPT_DIR/closed-end/First-Stage_Reasoning" \
  --output_dir     "$SLAKE_CKPT_DIR/closed-end" \
  --dataset        slake \
  --method         First-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

echo "[SLAKE-Closed] Step 5/5: Second-Stage Training..."
python closed_end_train.py \
  --train_text_file_path "$SLAKE_CKPT_DIR/closed-end/First-Stage_Reasoning/train.json" \
  --img_file_path        "$SLAKE_IMG_FEATURES" \
  --img_name_map         "$SLAKE_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$SLAKE_CKPT_DIR/closed-end" \
  --dataset              slake \
  --method               Second-Stage_Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

python closed_end_generate.py \
  --text_file_path "$SLAKE_CKPT_DIR/closed-end/First-Stage_Reasoning/test.json" \
  --img_file_path  "$SLAKE_IMG_FEATURES" \
  --img_name_map   "$SLAKE_NAME_MAP" \
  --model_path     "$SLAKE_CKPT_DIR/closed-end/Second-Stage_Reasoning" \
  --output_dir     "$SLAKE_CKPT_DIR/closed-end" \
  --dataset        slake \
  --method         Second-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

echo ""
echo "============================================================"
echo "  R-SLAKE-Bangla — Open-End"
echo "============================================================"

echo "[SLAKE-Open] Step 1/5: Training Explanation..."
python open_end_train.py \
  --train_text_file_path "$SLAKE_OPEN_DIR/train_bengali.json" \
  --img_file_path        "$SLAKE_IMG_FEATURES" \
  --img_name_map         "$SLAKE_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$SLAKE_CKPT_DIR/open-end" \
  --dataset              slake \
  --method               Explanation \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

echo "[SLAKE-Open] Step 2/5: Training without_R..."
python open_end_train.py \
  --train_text_file_path "$SLAKE_OPEN_DIR/train_bengali.json" \
  --img_file_path        "$SLAKE_IMG_FEATURES" \
  --img_name_map         "$SLAKE_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$SLAKE_CKPT_DIR/open-end" \
  --dataset              slake \
  --method               without_R \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

echo "[SLAKE-Open] Step 3/5: Training Reasoning..."
python open_end_train.py \
  --train_text_file_path "$SLAKE_OPEN_DIR/train_bengali.json" \
  --img_file_path        "$SLAKE_IMG_FEATURES" \
  --img_name_map         "$SLAKE_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$SLAKE_CKPT_DIR/open-end" \
  --dataset              slake \
  --method               Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

echo "[SLAKE-Open] Step 4/5: First-Stage Training..."
python open_end_train.py \
  --train_text_file_path "$SLAKE_OPEN_DIR/train_bengali.json" \
  --img_file_path        "$SLAKE_IMG_FEATURES" \
  --img_name_map         "$SLAKE_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$SLAKE_CKPT_DIR/open-end" \
  --dataset              slake \
  --method               First-Stage_Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM \
  --rational

mkdir -p "$SLAKE_CKPT_DIR/open-end/First-Stage_Reasoning"
python open_end_generate.py \
  --text_file_path "$SLAKE_OPEN_DIR/train_bengali.json" \
  --img_file_path  "$SLAKE_IMG_FEATURES" \
  --img_name_map   "$SLAKE_NAME_MAP" \
  --model_path     "$SLAKE_CKPT_DIR/open-end/First-Stage_Reasoning" \
  --output_dir     "$SLAKE_CKPT_DIR/open-end" \
  --dataset        slake \
  --method         First-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

python open_end_generate.py \
  --text_file_path "$SLAKE_OPEN_DIR/test_bengali.json" \
  --img_file_path  "$SLAKE_IMG_FEATURES" \
  --img_name_map   "$SLAKE_NAME_MAP" \
  --model_path     "$SLAKE_CKPT_DIR/open-end/First-Stage_Reasoning" \
  --output_dir     "$SLAKE_CKPT_DIR/open-end" \
  --dataset        slake \
  --method         First-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

echo "[SLAKE-Open] Step 5/5: Second-Stage Training..."
python open_end_train.py \
  --train_text_file_path "$SLAKE_CKPT_DIR/open-end/First-Stage_Reasoning/train.json" \
  --img_file_path        "$SLAKE_IMG_FEATURES" \
  --img_name_map         "$SLAKE_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$SLAKE_CKPT_DIR/open-end" \
  --dataset              slake \
  --method               Second-Stage_Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

python open_end_generate.py \
  --text_file_path "$SLAKE_CKPT_DIR/open-end/First-Stage_Reasoning/test.json" \
  --img_file_path  "$SLAKE_IMG_FEATURES" \
  --img_name_map   "$SLAKE_NAME_MAP" \
  --model_path     "$SLAKE_CKPT_DIR/open-end/Second-Stage_Reasoning" \
  --output_dir     "$SLAKE_CKPT_DIR/open-end" \
  --dataset        slake \
  --method         Second-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED


# =============================================================================
#  R-PathVQA-Bengali
# =============================================================================
echo ""
echo "============================================================"
echo "  R-PathVQA-Bengali — Closed-End"
echo "============================================================"

echo "[PathVQA-Closed] Step 1/5: Training Explanation..."
python closed_end_train.py \
  --train_text_file_path "$PATH_CLOSED_DIR/trainset_bengali.json" \
  --img_file_path        "$PATH_IMG_FEATURES" \
  --img_name_map         "$PATH_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$PATH_CKPT_DIR/closed-end" \
  --dataset              path \
  --method               Explanation \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

echo "[PathVQA-Closed] Step 2/5: Training without_R..."
python closed_end_train.py \
  --train_text_file_path "$PATH_CLOSED_DIR/trainset_bengali.json" \
  --img_file_path        "$PATH_IMG_FEATURES" \
  --img_name_map         "$PATH_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$PATH_CKPT_DIR/closed-end" \
  --dataset              path \
  --method               without_R \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

echo "[PathVQA-Closed] Step 3/5: Training Reasoning..."
python closed_end_train.py \
  --train_text_file_path "$PATH_CLOSED_DIR/trainset_bengali.json" \
  --img_file_path        "$PATH_IMG_FEATURES" \
  --img_name_map         "$PATH_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$PATH_CKPT_DIR/closed-end" \
  --dataset              path \
  --method               Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

echo "[PathVQA-Closed] Step 4/5: First-Stage Training..."
python closed_end_train.py \
  --train_text_file_path "$PATH_CLOSED_DIR/trainset_bengali.json" \
  --img_file_path        "$PATH_IMG_FEATURES" \
  --img_name_map         "$PATH_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$PATH_CKPT_DIR/closed-end" \
  --dataset              path \
  --method               First-Stage_Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM \
  --rational

mkdir -p "$PATH_CKPT_DIR/closed-end/First-Stage_Reasoning"
python closed_end_generate.py \
  --text_file_path "$PATH_CLOSED_DIR/trainset_bengali.json" \
  --img_file_path  "$PATH_IMG_FEATURES" \
  --img_name_map   "$PATH_NAME_MAP" \
  --model_path     "$PATH_CKPT_DIR/closed-end/First-Stage_Reasoning" \
  --output_dir     "$PATH_CKPT_DIR/closed-end" \
  --dataset        path \
  --method         First-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

python closed_end_generate.py \
  --text_file_path "$PATH_CLOSED_DIR/testset_bengali.json" \
  --img_file_path  "$PATH_IMG_FEATURES" \
  --img_name_map   "$PATH_NAME_MAP" \
  --model_path     "$PATH_CKPT_DIR/closed-end/First-Stage_Reasoning" \
  --output_dir     "$PATH_CKPT_DIR/closed-end" \
  --dataset        path \
  --method         First-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

echo "[PathVQA-Closed] Step 5/5: Second-Stage Training..."
python closed_end_train.py \
  --train_text_file_path "$PATH_CKPT_DIR/closed-end/First-Stage_Reasoning/train.json" \
  --img_file_path        "$PATH_IMG_FEATURES" \
  --img_name_map         "$PATH_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$PATH_CKPT_DIR/closed-end" \
  --dataset              path \
  --method               Second-Stage_Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

python closed_end_generate.py \
  --text_file_path "$PATH_CKPT_DIR/closed-end/First-Stage_Reasoning/test.json" \
  --img_file_path  "$PATH_IMG_FEATURES" \
  --img_name_map   "$PATH_NAME_MAP" \
  --model_path     "$PATH_CKPT_DIR/closed-end/Second-Stage_Reasoning" \
  --output_dir     "$PATH_CKPT_DIR/closed-end" \
  --dataset        path \
  --method         Second-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

echo ""
echo "============================================================"
echo "  R-PathVQA-Bengali — Open-End"
echo "============================================================"

echo "[PathVQA-Open] Step 1/5: Training Explanation..."
python open_end_train.py \
  --train_text_file_path "$PATH_OPEN_DIR/trainset_bengali.json" \
  --img_file_path        "$PATH_IMG_FEATURES" \
  --img_name_map         "$PATH_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$PATH_CKPT_DIR/open-end" \
  --dataset              path \
  --method               Explanation \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

echo "[PathVQA-Open] Step 2/5: Training without_R..."
python open_end_train.py \
  --train_text_file_path "$PATH_OPEN_DIR/trainset_bengali.json" \
  --img_file_path        "$PATH_IMG_FEATURES" \
  --img_name_map         "$PATH_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$PATH_CKPT_DIR/open-end" \
  --dataset              path \
  --method               without_R \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

echo "[PathVQA-Open] Step 3/5: Training Reasoning..."
python open_end_train.py \
  --train_text_file_path "$PATH_OPEN_DIR/trainset_bengali.json" \
  --img_file_path        "$PATH_IMG_FEATURES" \
  --img_name_map         "$PATH_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$PATH_CKPT_DIR/open-end" \
  --dataset              path \
  --method               Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

echo "[PathVQA-Open] Step 4/5: First-Stage Training..."
python open_end_train.py \
  --train_text_file_path "$PATH_OPEN_DIR/trainset_bengali.json" \
  --img_file_path        "$PATH_IMG_FEATURES" \
  --img_name_map         "$PATH_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$PATH_CKPT_DIR/open-end" \
  --dataset              path \
  --method               First-Stage_Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM \
  --rational

mkdir -p "$PATH_CKPT_DIR/open-end/First-Stage_Reasoning"
python open_end_generate.py \
  --text_file_path "$PATH_OPEN_DIR/trainset_bengali.json" \
  --img_file_path  "$PATH_IMG_FEATURES" \
  --img_name_map   "$PATH_NAME_MAP" \
  --model_path     "$PATH_CKPT_DIR/open-end/First-Stage_Reasoning" \
  --output_dir     "$PATH_CKPT_DIR/open-end" \
  --dataset        path \
  --method         First-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

python open_end_generate.py \
  --text_file_path "$PATH_OPEN_DIR/testset_bengali.json" \
  --img_file_path  "$PATH_IMG_FEATURES" \
  --img_name_map   "$PATH_NAME_MAP" \
  --model_path     "$PATH_CKPT_DIR/open-end/First-Stage_Reasoning" \
  --output_dir     "$PATH_CKPT_DIR/open-end" \
  --dataset        path \
  --method         First-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

echo "[PathVQA-Open] Step 5/5: Second-Stage Training..."
python open_end_train.py \
  --train_text_file_path "$PATH_CKPT_DIR/open-end/First-Stage_Reasoning/train.json" \
  --img_file_path        "$PATH_IMG_FEATURES" \
  --img_name_map         "$PATH_NAME_MAP" \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir           "$PATH_CKPT_DIR/open-end" \
  --dataset              path \
  --method               Second-Stage_Reasoning \
  --epoch                $EPOCH --bs $BS --lr $LR --wd $WD --seed $SEED \
  --source_len           $SOURCE_LEN --target_len $TARGET_LEN \
  --fp16 --grad_accum    $GRAD_ACCUM

python open_end_generate.py \
  --text_file_path "$PATH_CKPT_DIR/open-end/First-Stage_Reasoning/test.json" \
  --img_file_path  "$PATH_IMG_FEATURES" \
  --img_name_map   "$PATH_NAME_MAP" \
  --model_path     "$PATH_CKPT_DIR/open-end/Second-Stage_Reasoning" \
  --output_dir     "$PATH_CKPT_DIR/open-end" \
  --dataset        path \
  --method         Second-Stage_Reasoning \
  --source_len     $SOURCE_LEN --target_len $TARGET_LEN --eval_bs $EVAL_BS --seed $SEED

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================================"
