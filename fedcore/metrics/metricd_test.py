import torch
from sklearn.metrics import r2_score
from api_metric import calculate_metric, calculate_metrics
from nlp_metrics import NLPAccuracy, NLPPrecision, NLPRecall, NLPF1

# Создание тестовых данных для классификации и регрессии
target_regression = torch.tensor([3.0, 5.0, 2.5, 7.5])
predict_regression = torch.tensor([2.8, 5.1, 2.6, 7.3])

target_classification = torch.tensor([1, 0, 1, 1])
pred_labels_classification = torch.tensor([1, 0, 0, 1])

# ---------------------------- Проверка метрик для регрессии ----------------------------

print("Testing regression metrics:")

# Проверка всех метрик
metrics = ["mse", "rmse", "mae", "msle", "mape", "r2"]
for metric in metrics:
    result = calculate_metric(target_regression, predict_regression, metric)
    print(f"{metric.upper()}:", result)

# ---------------------------- Проверка метрик для классификации ----------------------------

print("\nTesting classification metrics:")

# Проверка классификационных метрик
classification_metrics = ["accuracy", "f1", "precision"]
for metric in classification_metrics:
    result = calculate_metric(target_classification, pred_labels_classification, metric)
    print(f"{metric.upper()}:", result)

# ---------------------------- Проверка специфичных метрик ----------------------------

print("\nTesting specific NLP metrics:")

# Используем числовые метки для классификационных метрик
nlp_target = [1, 0, 1, 0]  # настоящие метки классов (числа)
nlp_pred = [1, 0, 0, 1]    # предсказанные метки классов (числа)

accuracy = NLPAccuracy()
precision = NLPPrecision()
recall = NLPRecall()
f1 = NLPF1()

print(f"Accuracy: {accuracy.metric(nlp_target, nlp_pred)}")
print(f"Precision: {precision.metric(nlp_target, nlp_pred)}")
print(f"Recall: {recall.metric(nlp_target, nlp_pred)}")
print(f"F1: {f1.metric(nlp_target, nlp_pred)}")

print("\nAll tests completed successfully! ✅")