import torch
import argparse
import re
import os
import json
import numpy as np

# Force single GPU to prevent DataParallel StopIteration issues
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model import T5ForMultimodalGeneration
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from dataset import OpenMedVQADataset


def run_signature(_args):
    return {
        "script": "open_end_train.py",
        "dataset": _args.dataset,
        "method": _args.method,
        "train_file": os.path.basename(_args.train_text_file_path),
        "pretrained_model_path": _args.pretrained_model_path,
        "source_len": _args.source_len,
        "target_len": _args.target_len,
        "lr": _args.lr,
        "bs": _args.bs,
        "grad_accum": _args.grad_accum,
        "seed": _args.seed,
    }


def signatures_match(current, saved):
    return all(saved.get(key) == value for key, value in current.items())


def train_loop(_args):
    torch.manual_seed(_args.seed)
    np.random.seed(_args.seed)
    torch.backends.cudnn.deterministic = True

    # ✅ Fixed: patch_size as explicit kwarg, not positional arg
    model = T5ForMultimodalGeneration.from_pretrained(
        _args.pretrained_model_path,
        patch_size=(100, 256),
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
    )
    # ✅ Silence tied-weights warning
    model.config.tie_word_embeddings = False

    tokenizer = AutoTokenizer.from_pretrained(_args.pretrained_model_path)
    datacollator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    save_dir = os.path.join(_args.output_dir, _args.method)
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)

    config = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        eval_strategy="no",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=_args.lr,
        per_device_train_batch_size=_args.bs,
        weight_decay=_args.wd,
        num_train_epochs=_args.epoch,
        predict_with_generate=True,
        generation_max_length=_args.target_len,
        load_best_model_at_end=False,
        report_to=["none"],
        disable_tqdm=True,
        # ✅ Prevent exploding gradients during early multimodal training
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        fp16=_args.fp16,
        gradient_accumulation_steps=_args.grad_accum,
    )

    # Post-processing for ROUGE evaluation (First-Stage produces rationales)
    def postprocess_text(_preds, _labels):
        _preds  = [pred.strip()  for pred  in _preds]
        _labels = [label.strip() for label in _labels]
        # Regex-based sentence splitting for Bangla (dari ।, ?, !)
        _preds  = ["\n".join(s.strip() for s in re.split(r'[।?!]', pred)  if s.strip()) for pred  in _preds]
        _labels = ["\n".join(s.strip() for s in re.split(r'[।?!]', label) if s.strip()) for label in _labels]
        return _preds, _labels

    def compute_metrics_rougel(eval_preds):
        import evaluate
        metric = evaluate.load("rouge")
        preds, targets = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds   = np.where(preds   != -100, preds,   tokenizer.pad_token_id)
        targets = np.where(targets != -100, targets, tokenizer.pad_token_id)
        decoded_preds   = tokenizer.batch_decode(preds,   skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_targets)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_token_len"] = np.mean(prediction_lens)
        return result

    train_set = OpenMedVQADataset(
        _tokenizer=tokenizer,
        _text_file_path=_args.train_text_file_path,
        _img_file_path=_args.img_file_path,
        _img_name_map=_args.img_name_map,
        _method=_args.method,
        _source_len=_args.source_len,
        _target_len=_args.target_len,
        _dataset=_args.dataset
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=config,
        train_dataset=train_set,
        data_collator=datacollator,
        # ✅ Use ROUGE for First-Stage (generates rationales), None for answer-only stages
        compute_metrics=compute_metrics_rougel if _args.rational else None,
    )

    # ✅ Auto-detect latest checkpoint and resume (no manual flag needed)
    latest_ckpt = None
    signature_path = os.path.join(save_dir, "run_signature.json")
    current_signature = run_signature(_args)
    if os.path.isdir(save_dir):
        checkpoints = [d for d in os.listdir(save_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            can_resume = False
            if os.path.exists(signature_path):
                with open(signature_path, "r", encoding="utf-8") as f:
                    saved_signature = json.load(f)
                can_resume = signatures_match(current_signature, saved_signature)
                if not can_resume:
                    print("[AUTO-RESUME] Existing checkpoints do not match this command. Training from scratch.")
            elif _args.resume_without_metadata:
                can_resume = True
                print("[AUTO-RESUME] No run signature found, but --resume_without_metadata was passed.")
            else:
                print("[AUTO-RESUME] Existing checkpoints have no run signature. Training from scratch to avoid using a stale checkpoint.")

            if can_resume:
                latest_ckpt = os.path.join(save_dir, sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1])
                print(f"[AUTO-RESUME] Found matching checkpoint: {latest_ckpt}")
        else:
            print("[AUTO-RESUME] No checkpoint found. Training from scratch.")
    else:
        print("[AUTO-RESUME] Save dir does not exist yet. Training from scratch.")

    with open(signature_path, "w", encoding="utf-8") as f:
        json.dump(current_signature, f, ensure_ascii=False, indent=2)

    trainer.train(resume_from_checkpoint=latest_ckpt)
    with open(signature_path, "w", encoding="utf-8") as f:
        json.dump(current_signature, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_text_file_path', type=str, default='None')
    parser.add_argument('--img_file_path',        type=str, default='None')
    parser.add_argument('--img_name_map',         type=str, default='None')
    parser.add_argument('--pretrained_model_path',type=str, default='google/mt5-base')
    parser.add_argument('--output_dir',           type=str, default='None')
    parser.add_argument('--method', type=str, choices=["Explanation", "Reasoning", "First-Stage_Reasoning", "Second-Stage_Reasoning", "without_R"])
    parser.add_argument('--source_len', type=int,   default=512)
    parser.add_argument('--target_len', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=5e-4)
    parser.add_argument('--epoch',      type=int,   default=20)
    parser.add_argument('--bs',         type=int,   default=8)
    parser.add_argument('--wd',         type=float, default=1e-2)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--dataset',    type=str,   choices=['rad', 'slake', 'path'])
    parser.add_argument('--fp16',       action='store_true', help='Use mixed precision (half memory usage)')
    parser.add_argument('--grad_accum', type=int,   default=1, help='Gradient accumulation steps')
    parser.add_argument('--rational',   action='store_true', help='Use ROUGE metric if rationale is present (First-Stage)')
    parser.add_argument('--resume_without_metadata', action='store_true', help='Resume old checkpoints created before run_signature.json existed')
    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    train_loop(args)
