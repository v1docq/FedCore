import pytest 
import sys
correct_path = "/home/user/projects/FedCore/FedCore"
sys.path.insert(0, correct_path)
from typing import *
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

from fedcore.metrics.quality import MetricFactory, LOADED_METRICS, _NEED_TO_MINIMIZE


class TextPseudoDataset:
    """Pseudo dataset with text prompts and references for NLP metrics testing."""
    def __init__(self, prompts: List[str], references: List[str]):
        """
        Args:
            prompts: List of input prompts for the model
            references: List of reference texts (ground truth)
        """
        self.prompts = prompts
        self.references = references
        assert len(prompts) == len(references), "Prompts and references must have the same length"
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.prompts[idx], self.references[idx]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INITIAL_MODEL = 'arnir0/Tiny-LLM'

model = AutoModelForCausalLM.from_pretrained(INITIAL_MODEL)
model.to(DEVICE)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(INITIAL_MODEL)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

pseudo_prompts = [
    "The capital of France is",
    "Machine learning is",
    "Python is a programming language that"
]
pseudo_references = [
    "The capital of France is Paris",
    "Machine learning is a subset of artificial intelligence",
    "Python is a programming language that is widely used"
]

pseudo_dataset = TextPseudoDataset(pseudo_prompts, pseudo_references)

METRICS = ['bleu', 'rouge', 'meteor']

def generate_predictions(model, tokenizer, prompts: List[str], max_length: int = 30) -> List[str]:
    """Generate text predictions from the model for given prompts."""
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=False,  
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(generated_text)
    
    return predictions

@pytest.mark.parametrize('metric_name', METRICS)
def test_evaluate_metrics(metric_name):
    """Test that evaluate metrics can be loaded and compute correctly using model predictions."""
    assert metric_name is not None
    
    mtr = MetricFactory.get_evaluate(metric_name)
    
    assert mtr is not None
    
    expected_minimize = _NEED_TO_MINIMIZE.get(metric_name.upper(), False)
    assert mtr.need_to_minimize == expected_minimize, \
        f'{metric_name} should have need_to_minimize={expected_minimize}'

    predictions = generate_predictions(model, tokenizer, pseudo_dataset.prompts)
    references = pseudo_dataset.references
    
    result = mtr.metric(target=references, predict=predictions)
    assert result is not None, f'Metric {metric_name} returned None'
    assert isinstance(result, torch.Tensor), f'Metric {metric_name} should return torch.Tensor, got {type(result)}'
    assert result.numel() > 0, f'Metric {metric_name} returned empty tensor'