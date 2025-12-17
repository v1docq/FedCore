# examples/nlp_metrics_demo.py
from __future__ import annotations
from typing import List
from transformers import pipeline
from fedcore.metrics import get

# 1) маленький готовый сентимент-классификатор (CPU)
clf = pipeline(
    task="text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,          # CPU
    truncation=True
)

texts = [
    "I love this movie. It's brilliant!",
    "This was a terrible experience.",
    "Meh, it's okay I guess.",
    "Absolutely fantastic work!",
    "I wouldn't recommend it."
]

# получаем предсказанные метки 0/1
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
y_pred: List[int] = [label2id[p["label"]] for p in clf(texts)]

# игрушечные «истинные» метки для проверки (в реальном кейсе — из датасета)
y_true: List[int] = [1, 0, 1, 1, 0]

print("Preds:", y_pred)
print("True: ", y_true)

# 2) считаем метрики через API FedCore (наш реестр)
metric_params = {
    "accuracy":  {},                      # без average
    "precision": {"average": "binary"},
    "recall":    {"average": "binary"},
    "f1":        {"average": "binary"},
}

for name, params in metric_params.items():
    m = get(name)
    res = m.compute(y_true=y_true, y_pred=y_pred, **params)
    print(f"{name} -> {res}")

# 3) (опционально) показать текстовые метрики для генерации
#    только если стоят sacrebleu / rouge-score
try:
    pred_strs = ["a cat sits on the mat", "the dog runs fast"]
    ref_strs  = ["the cat is sitting on the mat", "a dog runs quickly"]

    bleu = get("sacrebleu")
    print("sacrebleu ->", bleu.compute(predictions=pred_strs, references=[[r] for r in ref_strs]))

    rouge = get("rouge")
    print("rouge ->", rouge.compute(predictions=pred_strs, references=ref_strs))
except Exception as e:
    print("Skip text metrics demo:", e)
