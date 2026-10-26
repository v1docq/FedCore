from itertools import product
import torch
from torch import nn
from fedcore.algorithm.quantization.catalogue import *
from fedcore.algorithm.quantization.graph_specs import *
from fedcore.algorithm.quantization.tensor_formats import *
from fedcore.algorithm.low_rank.allocation import Allocation,exhaustive_allocate,ScoreObjective
from fedcore.algorithm.quantization.observation import observe_operator_rows


def test_heterogeneous_catalogue_exact_allocation_loaded_full_evaluation():
    torch.manual_seed(22)
    model = nn.Sequential(nn.Linear(5,6),nn.Tanh(),nn.Linear(6,3))
    x = torch.randn(12,5)
    q = QuantizedTensorSpec(4,grid=Grid.ENDPOINT,rounding=Rounding.STOCHASTIC)
    alternatives = {"0":(TransformSpec(Method.DENSE),TransformSpec(Method.LPLR,rank=2,left=q,right=q)),
        "2":(TransformSpec(Method.PTQ,weight=QuantizedTensorSpec(2)),TransformSpec(Method.TENDER),
             TransformSpec(Method.CALDERA,rank=1,weight=QuantizedTensorSpec(4)))}
    catalogue = prepare_quantization_catalogue(model,alternatives,calibration=x)
    assert isinstance(catalogue,QuantizationCatalogue),catalogue
    assert catalogue.evidence["local_units"] == "output_squared_error"
    budget = max(sum(entry.physical_bytes for entry in pair) for pair in product(
        [e for e in catalogue.entries if e.path=="0"],[e for e in catalogue.entries if e.path=="2"]))
    allocated = allocate_quantization_catalogue(catalogue,budget)
    reference = exhaustive_allocate(catalogue.tables,budget,objective=ScoreObjective("local_quantization_error"),unit="bytes")
    assert allocated.selected == reference.selected
    assert allocated.total_cost == reference.total_cost
    y = model(x).detach()
    result = assess_quantization_selection(model,catalogue,allocated,calibration=x,
        evaluate=lambda loaded:float((loaded(x)-y).square().mean()),evaluation_context="heldout-toy-v1")
    assert isinstance(result,AssessedArtifact),result
    assert result.artifact.storage_bytes == allocated.total_cost
    assert result.whole_model_quality >= 0
    assert isinstance(allocate_quantization_catalogue(catalogue,0),BudgetInfeasible)
    stale = assess_quantization_selection(model,catalogue,allocated,calibration=x+1,
        evaluate=lambda _:0.,evaluation_context="other")
    assert isinstance(stale,StructuralInvalid)


def test_alias_scope_counted_once_and_true_child_input_observed():
    class Preprocess(nn.Module):
        def __init__(self):
            super().__init__(); self.add_module("0",nn.Linear(3,2))
        def forward(self,x):return self._modules["0"](x*7+3)
    m = Preprocess(); x = torch.randn(5,3)
    rows = observe_operator_rows(m,x,("0",),DEFAULT_MAX_PEAK_BYTES)
    assert torch.equal(rows["0"],x*7+3)
    layer = nn.Linear(3,2)
    model = nn.Sequential(layer,layer)
    # Widths do not compose, so provide explicit scope calibration.
    catalogue = prepare_quantization_catalogue(model,{"0":(TransformSpec(Method.PTQ),)},calibration={"0":x})
    assert isinstance(catalogue,QuantizationCatalogue),catalogue
    assert len(catalogue.tables)==1 and catalogue.tables[0].target_paths == ("0","1")
    allocation = allocate_quantization_catalogue(catalogue,100000)
    result = assess_quantization_selection(model,catalogue,allocation,calibration={"0":x},evaluate=lambda _:0.,evaluation_context="alias-test")
    assert isinstance(result,AssessedArtifact),result
    assert result.artifact.model[0] is result.artifact.model[1]
    assert result.artifact.storage_bytes == catalogue.entries[0].physical_bytes
