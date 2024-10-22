import torch

from chronos import ChronosPipeline


def chronos_small():
    model = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    return model
