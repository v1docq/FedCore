"""Independent data and MLM arithmetic checks, without pretrained downloads."""
from types import SimpleNamespace

import pytest

import torch
from torch import nn

pytest.importorskip('transformers',reason='BERT experiments require the llm extra')

from fedcore.experiments.protocol import TensorSplit
from fedcore.experiments.sparse_pretrained import (
    parse_articles, assign_articles, mask_windows, evaluate_mlm,
)


def test_top_heading_owns_subsections_and_duplicate_title_is_one_unit():
    articles = parse_articles([
        ' = First = \n', 'paragraph one\n', ' == Inner == \n', 'paragraph two\n',
        ' = Second = \n', 'another article\n', ' = First = \n', 'a second fragment\n',
    ], 'train')
    assert [a.title for a in articles] == ['first', 'second', 'first']
    assert 'paragraph two' in articles[0].text
    assert articles[0].article_id == articles[2].article_id
    assert articles[0].text_hash != articles[2].text_hash
    assigned = assign_articles(articles)
    assert assigned[0][1] == assigned[2][1]


def test_test_article_wins_overlap_even_if_train_was_read_first():
    articles = parse_articles([' = Same = \n','old paragraph\n'], 'train')
    articles += parse_articles([' = Same = \n','new paragraph\n'], 'validation')
    articles += parse_articles([' = Same = \n','test paragraph\n'], 'test')
    assert [role for _,role in assign_articles(articles)] == ['excluded_overlap','excluded_overlap','test']


def test_mask_plan_is_exact_without_replacement_and_leaves_special_positions():
    original = torch.arange(3*64).reshape(3,64)
    before = original.clone()
    x,y = mask_windows(original,999,107)
    assert torch.equal(original,before)
    assert torch.equal(x[:,[0,-1]],original[:,[0,-1]])
    assert torch.equal((y != -100).sum(1),torch.full((3,),9))
    assert torch.equal(x[y != -100],torch.full((27,),999))
    assert torch.equal(y[y != -100],original[y != -100])
    other_x,other_y = mask_windows(original,999,107)
    assert torch.equal(x,other_x) and torch.equal(y,other_y)


class LookupLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('table',torch.tensor([
            [-2.,1.,0.], [3.,0.,-1.], [.5,-.5,1.5],
        ],dtype=torch.float64))

    def forward(self,input_ids,attention_mask,token_type_ids):
        assert bool((attention_mask == 1).all())
        assert bool((token_type_ids == 0).all())
        return SimpleNamespace(logits=self.table[input_ids])


def test_mlm_loss_uses_vocabulary_axis_and_exact_masked_token_denominator():
    x = (torch.arange(2*64).reshape(2,64)%3).long()
    labels = torch.full_like(x,-100)
    labels[:,1:10] = (x[:,1:10]+1)%3
    split = TensorSplit(x,labels,('window_a','window_b'),('article_a','article_b'))
    model = LookupLogits()
    result,scores = evaluate_mlm(model,split,return_logits=True)
    expected_losses = []
    correct = 0
    for row in range(2):
        total = 0.
        for position in range(1,10):
            vector = model.table[int(x[row,position])]
            target = int(labels[row,position])
            total += float(torch.logsumexp(vector,0)-vector[target])
            correct += int(int(vector.argmax()) == target)
        expected_losses.append(total)
    assert result['masked_tokens'] == 18
    assert abs(result['loss_sum']-sum(expected_losses)) < 1e-12
    assert abs(result['mlm_nll']-sum(expected_losses)/18) < 1e-12
    assert result['top1_accuracy'] == correct/18
    assert all(abs(a-b/9) < 1e-12 for a,b in zip(result['window_nll'],expected_losses))
    torch.testing.assert_close(scores,model.table[x],atol=0,rtol=0)
