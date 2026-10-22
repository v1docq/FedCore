"""Independent numerical/provenance witnesses found during the P2 review."""
from copy import deepcopy

import torch
from torch import nn

from fedcore.algorithm.low_rank.method_execution import transform_method
from fedcore.algorithm.low_rank.method_specs import EoRA


def test_author_eora_snapshot_distinguishes_batch_dependent_statistics():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        generator = torch.Generator().manual_seed(204)
        reference = nn.Sequential(nn.Linear(3, 2, bias=False)).double().eval()
        with torch.no_grad():
            reference[0].weight.copy_(torch.randn((2, 3), generator=generator, dtype=torch.float64))
        base = deepcopy(reference)
        with torch.no_grad():
            base[0].weight.zero_()
        inputs = torch.randn((12, 3), generator=generator, dtype=torch.float64)
        spec = EoRA(gram_update_version="author_fixed_n_v1")

        def execute(batch_size):
            return transform_method(reference, inputs, spec, rank=1,
                                    target_paths=("0",), base_model=base, batch_size=batch_size)

        first, second, repeated = execute(3), execute(4), execute(3)
        a, b, again = (result.evidence["layers"][0] for result in (first, second, repeated))
        assert len(a["numerics"]["updates"]) == 4
        assert len(b["numerics"]["updates"]) == 3
        assert float((first.model(inputs)-second.model(inputs)).abs().max()) > 1e-6
        assert a["snapshot_id"] != b["snapshot_id"]
        assert a["snapshot_id"] == again["snapshot_id"]
        torch.testing.assert_close(first.model(inputs), repeated.model(inputs), rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)
