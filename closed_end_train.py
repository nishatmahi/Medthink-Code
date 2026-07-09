import evaluate
import argparse
import re
import os
import json
import numpy as np
import torch

# Force single GPU to prevent DataParallel StopIteration issues
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from dataset import ClosedMedVQADataset
from model import T5ForMultimodalGeneration


def run_signature(_args):
    return {
        "script": "closed_end_train.py",
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
    # FIX: torch.manual_seed alone does not seed the CUDA RNG
    torch.manual_seed(_args.seed)
    torch.cuda.manual_seed_all(_args.seed)
    np.random.seed(_args.seed)
    torch.backends.cudnn.deterministic = True

    # FIX: removed ignore_mismatched_sizes=True.
    # It silently re-randomizes embeddings/lm_head on a vocab mismatch
    # (e.g. banglat5 32100 vs mt5-base 250112). We want a loud crash instead.
    model = T5ForMultimodalGeneration.from_pretrained(
        _args.pretrained_model_path,
        patch_size=(100, 256),
        torch_dtype=torch.float32,
    )
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
        # FIX: was 1. With 1, the checkpoint you resumed from is deleted the
        # moment the next one is written. A disconnect mid-write leaves you
        # with a truncated checkpoint and nothing to fall back to.
        save_total_limit=2,
        learning_rate=_args.lr,
        per_device_train_batch_size=_args.bs,
        weight_decay=_args.wd,
        num_train_epochs=_args.epoch,
        predict_with_generate=True,
        generation_max_length=_args.target_len,
        load_best_model_at_end=False,
        report_to=["none"],
        disable_tqdm=True,
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        fp16=_args.fp16,
        bf16=_args.bf16,
        gradient_accumulation_steps=_args.grad_accum,
    )

    # ========== compute_metrics (currently unused: eval_strategy="no") ========
    def postprocess_text(_preds, _labels):
        _preds  = [pred.strip()  for pred  in _preds]
        _labels = [label.strip() for label in _labels]
        _preds  = ["\n".join(s.strip() for s in re.split(r'[।?!]', pred)  if s.strip()) for pred  in _preds]
        _labels = ["\n".join(s.strip() for s in re.split(r'[।?!]', label) if s.strip()) for label in _labels]
        return _preds, _labels

    def extract_ans(_ans):
        pattern = re.compile(r'সঠিক উত্তর হলো\s*\(([A-Z])\)')
        res = pattern.findall(_ans)
        return res[0] if len(res) == 1 else "FAILED"

    def compute_metrics_rougel(eval_preds):
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
        result["gen_token_len"] = np.mean(
            [np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]
        )
        return result

    def compute_metrics_acc(eval_preds):
        preds, targets = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds   = np.where(preds   != -100, preds,   tokenizer.pad_token_id)
        targets = np.where(targets != -100, targets, tokenizer.pad_token_id)
        preds   = tokenizer.batch_decode(preds,   skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        assert len(preds) == len(targets)
        correct = sum(extract_ans(targets[i]) == extract_ans(p) for i, p in enumerate(preds))
        return {'accuracy': correct / len(targets)}
    # =========================================================================

    train_set = ClosedMedVQADataset(
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
        compute_metrics=compute_metrics_rougel if _args.rational else compute_metrics_acc
    )

    # ---------------- Auto-detect latest checkpoint and resume ----------------
    latest_ckpt = None
    signature_path = os.path.join(save_dir, "run_signature.json")
    current_signature = run_signature(_args)

    # FIX: require isdir — a stray file named checkpoint-* used to crash the sort
    checkpoints = [
        d for d in os.listdir(save_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(save_dir, d))
    ] if os.path.isdir(save_dir) else []

    if checkpoints:
        can_resume = False
        if os.path.exists(signature_path):
            with open(signature_path, "r", encoding="utf-8") as f:
                saved_signature = json.load(f)
            can_resume = signatures_match(current_signature, saved_signature)
            if not can_resume:
                diff = {k: (saved_signature.get(k), v)
                        for k, v in current_signature.items()
                        if saved_signature.get(k) != v}
                print(f"[AUTO-RESUME] Signature mismatch (saved, current): {diff}")
        elif _args.resume_without_metadata:
            can_resume = True
            print("[AUTO-RESUME] No run signature found, but --resume_without_metadata was passed.")
        else:
            print("[AUTO-RESUME] Checkpoints have no run signature. Pass --resume_without_metadata to use them.")

        if can_resume:
            latest_ckpt = os.path.join(save_dir, sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1])
            print(f"[AUTO-RESUME] Resuming from: {latest_ckpt}")
        else:
            # FIX: never silently discard a long run. Make the user opt in.
            if not _args.force_restart:
                raise SystemExit(
                    f"\n[ABORT] {len(checkpoints)} checkpoint(s) exist in {save_dir} but cannot be "
                    f"resumed with these arguments.\nTraining from scratch would discard them.\n"
                    f"Pass --force_restart if that is what you want.\n"
                )
            print("[AUTO-RESUME] --force_restart passed. Training from scratch.")
    else:
        print("[AUTO-RESUME] No checkpoint found. Training from scratch.")

    with open(signature_path, "w", encoding="utf-8") as f:
        json.dump(current_signature, f, ensure_ascii=False, indent=2)

    trainer.train(resume_from_checkpoint=latest_ckpt)

    with open(signature_path, "w", encoding="utf-8") as f:
        json.dump(current_signature, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_text_file_path', type=str, default='None')
    parser.add_argument('--img_file_path', type=str, default='None')
    parser.add_argument('--img_name_map', type=str, default='None')
    parser.add_argument('--pretrained_model_path', type=str, default='None')
    parser.add_argument('--output_dir', type=str, default='None')
    parser.add_argument('--method', type=str, choices=["Explanation", "Reasoning", "First-Stage_Reasoning", "Second-Stage_Reasoning", "without_R"])
    parser.add_argument('--source_len', type=int,   default=512)
    parser.add_argument('--target_len', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=3e-5)
    parser.add_argument('--epoch',      type=int,   default=20)
    parser.add_argument('--bs',         type=int,   default=8)
    parser.add_argument('--wd',         type=float, default=1e-2)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--dataset',    type=str,   choices=['rad', 'slake', 'path'])
    parser.add_argument('--fp16',       action='store_true', help='UNSAFE for mT5 — pretrained in bf16, overflows to NaN. Use --bf16.')
    parser.add_argument('--bf16',       action='store_true', help='Mixed precision, safe for mT5. Needs Ampere+ (A100/L4). Not T4.')
    parser.add_argument('--grad_accum', type=int,   default=1, help='Gradient accumulation steps')
    parser.add_argument('--rational',   action='store_true', help='Use ROUGE metric if rational is present')
    parser.add_argument('--resume_without_metadata', action='store_true', help='Resume checkpoints created before run_signature.json existed')
    parser.add_argument('--force_restart', action='store_true', help='Discard existing checkpoints and train from scratch')
    args = parser.parse_args()

    if args.fp16 and args.bf16:
        raise SystemExit("[ABORT] Pass --fp16 or --bf16, not both.")
    if args.fp16:
        print("\n[WARNING] --fp16 with an mT5 model produces NaN loss. mT5 was pretrained "
              "in bfloat16. Use --bf16 on Ampere+ GPUs, or neither on a T4.\n")

    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    train_loop(args)
