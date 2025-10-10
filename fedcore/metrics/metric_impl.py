import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class TestMetrics:
    """Класс для тестирования всех метрик"""
    
    @staticmethod
    def create_test_data():
        """Создание тестовых данных для разных типов метрик"""
        
        # Данные для классификации
        classification_data = {
            'true': torch.tensor([0, 1, 1, 0, 1, 0, 0, 1]),
            'pred': torch.tensor([0, 1, 0, 0, 1, 1, 0, 1]),
            'probs': torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], 
                                 [0.8, 0.2], [0.1, 0.9], [0.4, 0.6],
                                 [0.7, 0.3], [0.3, 0.7]])
        }
        
        # Данные для регрессии
        regression_data = {
            'true': torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            'pred': torch.tensor([1.1, 1.9, 3.2, 3.8, 5.1])
        }
        
        # Данные для NLP
        nlp_data = {
            'references': ["This is a test sentence", "Another example text"],
            'predictions': ["This is test sentence", "Another text example"]
        }
        
        # Данные для distillation
        distillation_data = {
            'student_attentions': [torch.randn(2, 8, 10) for _ in range(3)],
            'teacher_attentions': [torch.randn(2, 8, 10) for _ in range(5)],
            'weights': torch.tensor([0.3, 0.4, 0.3]),
            'mapping': torch.tensor([0, 2, 4]),
            'student_feats': [torch.randn(2, 64) for _ in range(3)],
            'teacher_feats': [torch.randn(2, 64) for _ in range(3)],
            'student_logits': torch.randn(2, 10),
            'teacher_logits': torch.randn(2, 10),
            'weight': torch.tensor(1.0)
        }
        
        return classification_data, regression_data, nlp_data, distillation_data
    
    @staticmethod
    def create_dataloader():
        """Создание тестового DataLoader"""
        features = torch.randn(100, 10)
        targets = torch.randint(0, 2, (100,))
        dataset = TensorDataset(features, targets)
        return DataLoader(dataset, batch_size=32)
    
    def test_nlp_metrics(self):
        """Тестирование NLP метрик"""
        print("=== Testing NLP Metrics ===")
        
        try:
            from nlp_metrics import (
                NLPAccuracy, NLPPrecision, NLPRecall, NLPF1, 
                SacreBLEU, BLEU, ROUGE, METEOR, BERTScore
            )
            
            _, _, nlp_data, _ = self.create_test_data()
            
            # Тестирование классификационных метрик
            accuracy = NLPAccuracy()
            precision = NLPPrecision()
            recall = NLPRecall()
            f1 = NLPF1()
            
            true_labels = [0, 1, 1, 0, 1]
            pred_labels = [0, 1, 0, 0, 1]
            
            try:
                acc_result = accuracy.compute(y_true=true_labels, y_pred=pred_labels)
                print(f"Accuracy: {acc_result}")
            except Exception as e:
                print(f"Accuracy error: {e}")
            
            try:
                prec_result = precision.compute(y_true=true_labels, y_pred=pred_labels)
                print(f"Precision: {prec_result}")
            except Exception as e:
                print(f"Precision error: {e}")
            
            try:
                rec_result = recall.compute(y_true=true_labels, y_pred=pred_labels)
                print(f"Recall: {rec_result}")
            except Exception as e:
                print(f"Recall error: {e}")
            
            try:
                f1_result = f1.compute(y_true=true_labels, y_pred=pred_labels)
                print(f"F1: {f1_result}")
            except Exception as e:
                print(f"F1 error: {e}")
            
            # Тестирование текстовых метрик
            bleu = SacreBLEU()
            rouge = ROUGE()
            
            try:
                bleu_result = bleu.compute(references=nlp_data['references'], 
                                         predictions=nlp_data['predictions'])
                print(f"BLEU: {bleu_result}")
            except Exception as e:
                print(f"BLEU error: {e}")
            
            try:
                rouge_result = rouge.compute(references=nlp_data['references'], 
                                           predictions=nlp_data['predictions'])
                print(f"ROUGE: {rouge_result}")
            except Exception as e:
                print(f"ROUGE error: {e}")
                
        except ImportError as e:
            print(f"NLP metrics import error: {e}")
        
        print()
    
    def test_cv_metrics(self):
        """Тестирование CV метрик"""
        print("=== Testing CV Metrics ===")
        
        try:
            from cv_metrics import (
                Throughput as CVThroughput, 
                Latency as CVLatency, 
                CV_quality_metric,
                IntermediateAttention, IntermediateFeatures, LastLayer
            )
            
            # Создание простой модели для тестирования
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 2)
            )
            
            dataloader = self.create_dataloader()
            
            # Тестирование метрик производительности
            try:
                throughput = CVThroughput.metric(model=model, dataset=dataloader)
                print(f"Throughput: {throughput}")
            except Exception as e:
                print(f"Throughput error: {e}")
            
            try:
                latency = CVLatency.metric(model=model, dataset=dataloader)
                print(f"Latency: {latency}")
            except Exception as e:
                print(f"Latency error: {e}")
            
            try:
                quality = CV_quality_metric.metric(model=model, dataset=dataloader)
                print(f"Quality metrics: {quality}")
            except Exception as e:
                print(f"Quality metrics error: {e}")
                
        except ImportError as e:
            print(f"CV metrics import error: {e}")
        
        print()
    
    def test_distillation_metrics(self):
        """Тестирование метрик дистилляции"""
        print("=== Testing Distillation Metrics ===")
        
        try:
            from cv_metrics import (
                IntermediateAttention, IntermediateFeatures, LastLayer
            )
            
            _, _, _, distillation_data = self.create_test_data()
            
            try:
                attention_loss = IntermediateAttention.metric(
                    student_attentions=distillation_data['student_attentions'],
                    teacher_attentions=distillation_data['teacher_attentions'],
                    weights=distillation_data['weights'],
                    student_teacher_attention_mapping=distillation_data['mapping']
                )
                print(f"Attention loss: {attention_loss}")
            except Exception as e:
                print(f"Attention loss error: {e}")
            
            try:
                feature_loss = IntermediateFeatures.metric(
                    student_feats=distillation_data['student_feats'],
                    teacher_feats=distillation_data['teacher_feats'],
                    weights=distillation_data['weights']
                )
                print(f"Feature loss: {feature_loss}")
            except Exception as e:
                print(f"Feature loss error: {e}")
            
            try:
                logit_loss = LastLayer.metric(
                    student_logits=distillation_data['student_logits'],
                    teacher_logits=distillation_data['teacher_logits'],
                    weight=distillation_data['weight']
                )
                print(f"Logit loss: {logit_loss}")
            except Exception as e:
                print(f"Logit loss error: {e}")
                
        except ImportError as e:
            print(f"Distillation metrics import error: {e}")
        
        print()
    
    def test_quality_metrics(self):
        """Тестирование quality метрик"""
        print("=== Testing Quality Metrics ===")
        
        try:
            from metric_impl import (
                RMSE, MSE, MSLE, MAE, MAPE, SMAPE, R2,
                Accuracy, Precision, F1, Logloss, ROCAUC,
                MASE
            )
            
            # Создаем функциональные обертки для тестирования
            def mape(target, predict):
                return MAPE.metric(target, predict)

            def mase(target, predict, seasonal_period=1):
                return MASE.metric(target, predict, seasonal_period)

            def smape(target, predict):
                return SMAPE.metric(target, predict)
            
            classification_data, regression_data, _, _ = self.create_test_data()
            
            # Тестирование регрессионных метрик
            try:
                rmse = RMSE.metric(regression_data['true'], regression_data['pred'])
                print(f"RMSE: {rmse}")
            except Exception as e:
                print(f"RMSE error: {e}")
            
            try:
                mse = MSE.metric(regression_data['true'], regression_data['pred'])
                print(f"MSE: {mse}")
            except Exception as e:
                print(f"MSE error: {e}")
            
            try:
                mae = MAE.metric(regression_data['true'], regression_data['pred'])
                print(f"MAE: {mae}")
            except Exception as e:
                print(f"MAE error: {e}")
            
            try:
                mape_val = MAPE.metric(regression_data['true'], regression_data['pred'])
                print(f"MAPE: {mape_val}")
            except Exception as e:
                print(f"MAPE error: {e}")
            
            try:
                smape_val = SMAPE.metric(regression_data['true'], regression_data['pred'])
                print(f"SMAPE: {smape_val}")
            except Exception as e:
                print(f"SMAPE error: {e}")
            
            try:
                r2 = R2.metric(regression_data['true'], regression_data['pred'])
                print(f"R2: {r2}")
            except Exception as e:
                print(f"R2 error: {e}")
            
            # Тестирование классификационных метрик
            try:
                accuracy = Accuracy.metric(classification_data['true'], classification_data['pred'])
                print(f"Classification Accuracy: {accuracy}")
            except Exception as e:
                print(f"Classification Accuracy error: {e}")
            
            try:
                precision = Precision.metric(classification_data['true'], classification_data['pred'])
                print(f"Classification Precision: {precision}")
            except Exception as e:
                print(f"Classification Precision error: {e}")
            
            try:
                f1_score = F1.metric(classification_data['true'], classification_data['pred'])
                print(f"Classification F1: {f1_score}")
            except Exception as e:
                print(f"Classification F1 error: {e}")
            
            try:
                logloss = Logloss.metric(classification_data['true'], classification_data['probs'])
                print(f"LogLoss: {logloss}")
            except Exception as e:
                print(f"LogLoss error: {e}")
            
            try:
                roc_auc = ROCAUC.metric(classification_data['true'], classification_data['probs'])
                print(f"ROC AUC: {roc_auc}")
            except Exception as e:
                print(f"ROC AUC error: {e}")
            
            # Тестирование функциональных метрик
            try:
                mape_func = mape(regression_data['true'], regression_data['pred'])
                print(f"Functional MAPE: {mape_func}")
            except Exception as e:
                print(f"Functional MAPE error: {e}")
            
            try:
                mase_val = mase(regression_data['true'], regression_data['pred'], seasonal_period=1)
                print(f"MASE: {mase_val}")
            except Exception as e:
                print(f"MASE error: {e}")
                
        except ImportError as e:
            print(f"Quality metrics import error: {e}")
        
        print()
    
    def test_api_metrics(self):
        """Тестирование API метрик"""
        print("=== Testing API Metrics ===")
        
        try:
            from api_metric import Throughput, Latency, CV_quality_metric as APICVQuality
            
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 2)
            )
            
            dataloader = self.create_dataloader()
            
            try:
                throughput = Throughput.metric(model=model, dataset=dataloader)
                print(f"API Throughput: {throughput}")
            except Exception as e:
                print(f"API Throughput error: {e}")
            
            try:
                latency = Latency.metric(model=model, dataset=dataloader)
                print(f"API Latency: {latency}")
            except Exception as e:
                print(f"API Latency error: {e}")
            
            try:
                quality = APICVQuality.metric(model=model, dataset=dataloader)
                print(f"API Quality metrics: {quality}")
            except Exception as e:
                print(f"API Quality metrics error: {e}")
                
        except ImportError as e:
            print(f"API metrics import error: {e}")
        
        print()
    
    def run_all_tests(self):
        """Запуск всех тестов"""
        print("Starting metric tests...\n")
        
        self.test_nlp_metrics()
        self.test_cv_metrics()
        self.test_distillation_metrics()
        self.test_quality_metrics()
        self.test_api_metrics()
        
        print("All tests completed!")

def main():
    """Основная функция для запуска тестов"""
    tester = TestMetrics()
    tester.run_all_tests()

if __name__ == "__main__":
    main()