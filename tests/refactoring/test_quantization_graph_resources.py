"""Ambient process RSS is part of the common G5 admission floor, not payload size."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.candidate_storage import CandidateStorageError,storage_guard
from fedcore.algorithm.low_rank.compression_composition import CompositionRequest
from fedcore.algorithm.quantization.graph_specs import (
    DEFAULT_MAX_PEAK_BYTES,QuantizationGraphRequest,OperatorRecipe,TransformSpec,Method,
)
from fedcore.algorithm.quantization.graph_execution import compress_quantization_graph,ExportVerifiedArtifact
from fedcore.algorithm.quantization.catalogue import prepare_quantization_catalogue,QuantizationCatalogue
from fedcore.algorithm.quantization.grammar import prepare_quantization_space,generate_genotype,compile_genotype
from fedcore.algorithm.quantization.program import quantize,QuantizationProgram,ProgramArtifact
from fedcore.algorithm.quantization.resources import StageBudgets
from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure


def _rss(monkeypatch,bytes):
    import psutil
    monkeypatch.setattr(psutil,'Process',lambda *args,**kwargs:SimpleNamespace(
        memory_info=lambda:SimpleNamespace(rss=bytes)))


@pytest.mark.parametrize('entrypoint',['graph','catalogue','program'])
def test_common_default_admits_high_ambient_rss_but_explicit_limit_is_strict(monkeypatch,entrypoint):
    # No large allocation: reproduce a long-lived CI worker with 2 GiB ambient RSS.
    model=nn.Sequential(nn.Linear(3,3));x=torch.ones(2,3)
    before={key:value.clone() for key,value in model.state_dict().items()}
    _rss(monkeypatch,2*1024**3)
    request=QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.PTQ)),))
    assert request.max_peak_bytes==CompositionRequest(()).max_peak_bytes==StageBudgets().max_rss_bytes==12*1024**3
    if entrypoint=='graph':
        admitted=compress_quantization_graph(model,request,calibration=x)
        refused=compress_quantization_graph(model,replace(request,max_peak_bytes=1024**3),calibration=x)
        assert type(admitted) is ExportVerifiedArtifact,admitted
    elif entrypoint=='catalogue':
        admitted=prepare_quantization_catalogue(model,{'0':(TransformSpec(Method.PTQ),)},calibration=x)
        refused=prepare_quantization_catalogue(model,{'0':(TransformSpec(Method.PTQ),)},calibration=x,max_peak_bytes=1024**3)
        assert type(admitted) is QuantizationCatalogue,admitted
        assert admitted.max_peak_bytes==DEFAULT_MAX_PEAK_BYTES
    else:
        admitted=quantize(model,QuantizationProgram(request),calibration=x)
        refused=quantize(model,QuantizationProgram(replace(request,max_peak_bytes=1024**3)),calibration=x)
        assert type(admitted) is ProgramArtifact,admitted
    assert type(refused) is ResourceFailure,refused
    assert refused.code=='ResourceLimitExceeded'
    for key,value in before.items():assert torch.equal(model.state_dict()[key],value)


def test_explicit_limit_retains_the_shared_named_buffer_reserve(monkeypatch):
    _rss(monkeypatch,1024**3-32*1024**2)
    # RSS alone fits; the existing shared 64 MiB reserve still makes it fail.
    with pytest.raises(CandidateStorageError):storage_guard(0,1024**3)
    model=nn.Sequential(nn.Linear(2,2))
    request=QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.PTQ)),),max_peak_bytes=1024**3)
    result=compress_quantization_graph(model,request)
    assert type(result) is ResourceFailure and result.code=='ResourceLimitExceeded'


def test_default_limit_is_still_enforced_and_space_preserves_explicit_policy(monkeypatch):
    model=nn.Sequential(nn.Linear(2,2));x=torch.ones(2,2)
    alternatives={'0':(TransformSpec(Method.PTQ),)}
    space=prepare_quantization_space(model,alternatives,calibration=x,evaluation_context='rss-limit-test')
    genotype=generate_genotype(space)
    default=compile_genotype(genotype,space)
    limited=compile_genotype(genotype,replace(space,max_peak_bytes=1024**3))
    assert default.request.max_peak_bytes==DEFAULT_MAX_PEAK_BYTES
    assert limited.request.max_peak_bytes==1024**3
    assert default.fingerprint!=limited.fingerprint
    _rss(monkeypatch,DEFAULT_MAX_PEAK_BYTES)
    result=compress_quantization_graph(model,default.request,calibration=x)
    assert type(result) is ResourceFailure and result.code=='ResourceLimitExceeded'
