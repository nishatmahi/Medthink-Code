import os
import argparse
import torch
import json
import numpy as np

# Force single GPU to prevent DataParallel StopIteration issues
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataset import ClosedMedVQADataset
from model import T5ForMultimodalGeneration
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

def eval_loop(_args):
    torch.manual_seed(_args.seed)
    np.random.seed(_args.seed)
    torch.backends.cudnn.deterministic = True

    # ✅ Fixed: patch_size as explicit kwarg, not positional arg
    model = T5ForMultimodalGeneration.from_pretrained(
        _args.model_path,
        patch_size=(100, 256),
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
    )
    # ✅ Silence tied-weights warning
    model.config.tie_word_embeddings = False

    tokenizer = AutoTokenizer.from_pretrained(_args.model_path)
    datacollator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    config = Seq2SeqTrainingArguments(
        output_dir="./",
        per_device_eval_batch_size=_args.eval_bs,
        predict_with_generate=True,
        generation_max_length=_args.target_len,
    )

    # Disable use_cache to avoid transformers v4.41+ T5 cross-attention key_length mismatch bug
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    model.config.use_cache = False

    trainer = Seq2SeqTrainer(
        model=model,
        args=config,
        data_collator=datacollator
    )

    data_set = ClosedMedVQADataset(
        _tokenizer=tokenizer,
        _text_file_path=_args.text_file_path,
        _img_file_path=_args.img_file_path,
        _img_name_map=_args.img_name_map,
        _method=_args.method,
        _source_len=_args.source_len,
        _target_len=_args.target_len,
        _dataset=_args.dataset
    )

    predictions = trainer.predict(test_dataset=data_set, max_length=256)
    preds, targets = predictions.predictions, predictions.label_ids

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    preds_text = [p.strip() for p in preds_text]

    problem_ids = data_set.problem_id
    questions_dict = {
        f"question_{problem_id}": preds_text[index]
        for index, problem_id in enumerate(problem_ids)
    }

    if _args.method == "First-Stage_Reasoning":
        with open(_args.text_file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        for key, value in questions_dict.items():
            question_number = key.split('_')[1]
            if question_number in raw_data:
                raw_data[question_number]['solution'] = value
            else:
                raise ValueError(f"Key {question_number} not found in raw data")
        save_path = os.path.join(
            _args.output_dir, _args.method,
            "test.json" if "test" in _args.text_file_path else "train.json"
        )
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=4)
    else:
        save_path = os.path.join(_args.output_dir, _args.method, "test_response.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(questions_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file_path', type=str, default='None')
    parser.add_argument('--img_file_path',  type=str, default='None')
    parser.add_argument('--img_name_map',   type=str, default='None')
    parser.add_argument('--model_path',     type=str, default='None')
    parser.add_argument('--output_dir',     type=str, default='None')
    parser.add_argument('--source_len',     type=int, default=512)
    parser.add_argument('--target_len',     type=int, default=256)
    parser.add_argument('--eval_bs',        type=int, default=8)
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--dataset', type=str, choices=['rad', 'slake'])
    parser.add_argument('--method',  type=str, choices=["Explanation", "Reasoning", "First-Stage_Reasoning", "Second-Stage_Reasoning", "without_R"])
    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    eval_loop(args)
