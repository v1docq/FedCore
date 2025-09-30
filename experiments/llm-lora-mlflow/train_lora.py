# train_lora.py — Windows-friendly, transformers 4.55.x (eval_strategy), LoRA + MLflow
import os
import json
import argparse
import inspect
from typing import List, Dict

import torch
import numpy as np
import mlflow
import evaluate
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    TrainingArguments as _TA,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model


# ---------- utils ----------

def cuda_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8  # Ampere+

def normalize_text(s: str) -> str:
    import re, string
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def f1_em(pred: str, golds: List[str]) -> Dict[str, float]:
    from collections import Counter
    def _f1(p, g):
        p_tokens, g_tokens = p.split(), g.split()
        common = Counter(p_tokens) & Counter(g_tokens)
        num_same = sum(common.values())
        if len(p_tokens) == 0 or len(g_tokens) == 0:
            return float(p_tokens == g_tokens)
        if num_same == 0:
            return 0.0
        precision = num_same / len(p_tokens)
        recall = num_same / len(g_tokens)
        return 2 * precision * recall / (precision + recall)

    norm_golds = [normalize_text(g) for g in golds if g]
    norm_pred = normalize_text(pred)
    em = float(norm_pred in norm_golds)
    f1 = max(_f1(norm_pred, g) for g in norm_golds) if norm_golds else 0.0
    return {"em": em, "f1": f1}

def make_prompt_and_answer(ex, task: str):
    if task == "summarization":
        prompt = (
            "Summarize the following dialogue.\n\n"
            f"Dialogue:\n{ex['dialogue']}\n\n"
            "### Summary:"
        )
        answer = ex["summary"]
    elif task == "qa":
        prompt = (
            "Read the context and answer the question.\n\n"
            f"Context: {ex['context']}\n\n"
            f"Question: {ex['question']}\n\n"
            "Answer:"
        )
        answer_list = ex.get("answers", {}).get("text", [])
        answer = answer_list[0] if len(answer_list) else ""
    else:
        raise ValueError("Unknown task")
    return prompt, answer

def build_tokenize_fn(tokenizer, task: str, max_input_len: int, max_target_len: int):
    def _tok(ex):
        prompt, answer = make_prompt_and_answer(ex, task)
        p_ids = tokenizer(prompt, truncation=True, max_length=max_input_len, add_special_tokens=True)["input_ids"]
        a_ids = tokenizer(
            answer + tokenizer.eos_token,
            truncation=True, max_length=max_target_len, add_special_tokens=False
        )["input_ids"]
        input_ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids
        return {"input_ids": input_ids, "labels": labels}
    return _tok

def guess_target_modules(model):
    candidates = {
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
        "wi","wo","w1","w2","w3","dense","fc_in","fc_out","proj"
    }
    present = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates:
            present.add(leaf)
    if not present:
        present = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
    return sorted(present)

# transformers 4.55.x: используем eval_strategy
def build_training_args(**kw):
    sig = inspect.signature(_TA.__init__)
    allowed = {k: v for k, v in kw.items() if k in sig.parameters}
    # fallback: если в другой версии ключ назывался иначе
    if "evaluation_strategy" in kw and "evaluation_strategy" not in sig.parameters and "eval_strategy" in sig.parameters:
        allowed["eval_strategy"] = kw["evaluation_strategy"]
    return _TA(**allowed)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--task", type=str, choices=["qa","summarization"], required=True)
    ap.add_argument("--dataset", type=str, choices=["squad","samsum"], required=True)
    ap.add_argument("--experiment", type=str, default="LLM-LoRA")
    ap.add_argument("--run_name", type=str, default=None)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # train hp
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--eval_steps", type=int, default=200)

    # lengths
    ap.add_argument("--max_input_len", type=int, default=1024)
    ap.add_argument("--max_target_len", type=int, default=256)

    # speed-up flags
    ap.add_argument("--train_samples", type=int, default=0, help="если >0, брать только первые N train")
    ap.add_argument("--eval_samples", type=int, default=1000, help="если >0, ограничить eval N примерами")

    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--tracking_uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI", "file:mlruns"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- mlflow ---
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    run_name = args.run_name or f"{args.task}_{args.dataset}_{args.model_id.split('/')[-1]}"

    with mlflow.start_run(run_name=run_name):
        if torch.cuda.is_available():
            mlflow.set_tag("gpu", torch.cuda.get_device_name(0))
        mlflow.log_params({
            "model_id": args.model_id, "task": args.task, "dataset": args.dataset,
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "max_input_len": args.max_input_len, "max_target_len": args.max_target_len
        })

        # --- data ---
        ds_name = "knkarthick/samsum" if args.dataset == "samsum" else "rajpurkar/squad"
        raw = load_dataset(ds_name)

        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        tok_fn = build_tokenize_fn(tokenizer, args.task, args.max_input_len, args.max_target_len)
        tokenized = raw.map(tok_fn, remove_columns=raw["train"].column_names)

        train_ds = tokenized["train"]
        if args.train_samples and args.train_samples > 0:
            train_ds = train_ds.select(range(min(args.train_samples, len(train_ds))))

        eval_split = "validation" if "validation" in tokenized else "test"
        eval_ds = tokenized[eval_split]
        if args.eval_samples and args.eval_samples > 0:
            eval_ds = eval_ds.select(range(min(args.eval_samples, len(eval_ds))))

        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

        # --- model ---
        torch_dtype = torch.bfloat16 if cuda_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)

        cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=cfg,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        target_modules = guess_target_modules(model)
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=target_modules, task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_cfg)

        # --- training ---
        train_args = build_training_args(
            output_dir=os.path.join(args.out_dir, "checkpoints"),
            do_train=True,
            do_eval=True,
            eval_strategy="steps",              # для 4.55.x
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_steps=args.warmup_steps,
            logging_steps=50,
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps,
            save_total_limit=2,
            report_to=["mlflow"],
            bf16=(torch_dtype == torch.bfloat16),
            fp16=(torch_dtype == torch.float16),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        trainer.train()

        # --- post-eval: генерация и метрики ---
        sample_raw = raw[eval_split].select(range(min(300, len(raw[eval_split]))))
        gen_kwargs = dict(max_new_tokens=256, do_sample=False, num_beams=4)

        preds, refs = [], []
        for ex in sample_raw:
            prompt, answer = make_prompt_and_answer(ex, args.task)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = text.split("### Summary:")[-1].strip() if args.task == "summarization" else text.split("Answer:")[-1].strip()
            preds.append(pred)
            if args.task == "summarization":
                refs.append(ex["summary"])
            else:
                refs.append(ex.get("answers", {}).get("text", [""]))

        if args.task == "summarization":
            rouge = evaluate.load("rouge")
            r = rouge.compute(predictions=preds, references=refs)
            mlflow.log_metrics({f"rouge_{k}": float(v) for k, v in r.items()})
            with open(os.path.join(args.out_dir, "rouge_sample.json"), "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2)
            mlflow.log_artifact(os.path.join(args.out_dir, "rouge_sample.json"))
        else:
            ems, f1s = [], []
            for pred, goldlist in zip(preds, refs):
                sc = f1_em(pred, goldlist)
                ems.append(sc["em"]); f1s.append(sc["f1"])
            mlflow.log_metrics({"EM": float(np.mean(ems)), "F1": float(np.mean(f1s))})

        # --- learning curves png ---
        hist = trainer.state.log_history
        steps_tr, tr_loss, steps_ev, ev_loss = [], [], [], []
        for h in hist:
            if "loss" in h and "epoch" in h:
                steps_tr.append(h.get("step", len(steps_tr))); tr_loss.append(h["loss"])
            if "eval_loss" in h:
                steps_ev.append(h.get("step", len(steps_ev))); ev_loss.append(h["eval_loss"])
        if tr_loss or ev_loss:
            plt.figure()
            if tr_loss: plt.plot(steps_tr, tr_loss, label="train_loss")
            if ev_loss: plt.plot(steps_ev, ev_loss, label="eval_loss")
            plt.xlabel("step"); plt.ylabel("loss"); plt.title("Learning curves"); plt.legend()
            path = os.path.join(args.out_dir, "learning_curves.png")
            plt.savefig(path, dpi=160, bbox_inches="tight")
            mlflow.log_artifact(path)

        # --- save LoRA adapter ---
        adapter_dir = os.path.join(args.out_dir, "lora_adapter")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        with open(os.path.join(adapter_dir, "BASE_MODEL.txt"), "w", encoding="utf-8") as f:
            f.write(args.model_id + "\n")
        mlflow.log_artifacts(adapter_dir, artifact_path="lora_adapter")

        # --- save run config ---
        cfg_path = os.path.join(args.out_dir, "run_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
        mlflow.log_artifact(cfg_path)


if __name__ == "__main__":
    main()
