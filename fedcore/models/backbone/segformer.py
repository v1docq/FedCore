from typing import Dict

import torch
from transformers import SegformerForSemanticSegmentation


def segformer_pretrain(id2label: Dict, label2id: Dict, pretrain_path: str = None):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    if pretrain_path:
        model = torch.load(pretrain_path, map_location="cpu")["model"]

    return model
