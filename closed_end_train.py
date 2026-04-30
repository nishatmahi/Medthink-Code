import nltk
import evaluate
import argparse
import re
import os
import numpy as np
import torch

# Force single GPU to prevent DataParallel StopIteration issues
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from dataset import ClosedMedVQADataset
from model import T5ForMultimodalGeneration

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

    config = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="no",
        save_total_limit=1,
        learning_rate=_args.lr,
        per_device_train_batch_size=_args.bs,
        weight_decay=_args.wd,
        num_train_epochs=_args.epoch,
        metric_for_best_model="rougeL" if _args.method == "First-Stage_Reasoning" else "accuracy",
        predict_with_generate=True,
        generation_max_length=_args.target_len,
        load_best_model_at_end=False,
        report_to=["none"],
        disable_tqdm=True,
        # ✅ Prevent exploding gradients during early multimodal training
        max_grad_norm=1.0,
        warmup_ratio=0.05,
    )

    # ========== Define compute_metrics functions ==============================
    def postprocess_text(_preds, _labels):
        _preds  = [pred.strip()  for pred  in _preds]
        _labels = [label.strip() for label in _labels]
        _preds  = ["\n".join(nltk.sent_tokenize(pred))  for pred  in _preds]
        _labels = ["\n".join(nltk.sent_tokenize(label)) for label in _labels]
        return _preds, _labels

    def extract_ans(_ans):
        pattern = re.compile(r'The answer is \(([A-Z])\)')
        res = pattern.findall(_ans)
        if len(res) == 1:
            return res[0]
        return "FAILED"

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
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_token_len"] = np.mean(prediction_lens)
        return result

    def compute_metrics_acc(eval_preds):
        preds, targets = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds   = np.where(preds   != -100, preds,   tokenizer.pad_token_id)
        targets = np.where(targets != -100, targets, tokenizer.pad_token_id)
        preds   = tokenizer.batch_decode(preds,   skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct = 0
        assert len(preds) == len(targets)
        for idx, pred in enumerate(preds):
            if extract_ans(targets[idx]) == extract_ans(pred):
                correct += 1
        return {'accuracy': 1.0 * correct / len(targets)}
    # ========== Define compute_metrics functions ==============================

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

    trainer.train()
    trainer.save_model(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_text_file_path', type=str, default='None')
    parser.add_argument('--img_file_path',        type=str, default='None')
    parser.add_argument('--img_name_map',         type=str, default='None')
    parser.add_argument('--pretrained_model_path',type=str, default='None')
    parser.add_argument('--output_dir',           type=str, default='None')
    parser.add_argument('--method', type=str, choices=["Explanation", "Reasoning", "First-Stage_Reasoning", "Second-Stage_Reasoning", "without_R"])
    parser.add_argument('--source_len', type=int,   default=512)
    parser.add_argument('--target_len', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=3e-5)   # ✅ lowered from 5e-5
    parser.add_argument('--epoch',      type=int,   default=20)
    parser.add_argument('--bs',         type=int,   default=8)
    parser.add_argument('--wd',         type=float, default=1e-2)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--dataset',    type=str,   choices=['rad', 'slake'])
    parser.add_argument('--rational',   action='store_true', help='Use ROUGE metric if rational is present')
    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    train_loop(args)
