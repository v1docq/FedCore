import os
import ctypes
import uuid
from types import SimpleNamespace
import pytest
import torch
from fedcore.experiments import heterogeneous_device as device
from fedcore.experiments.heterogeneous_deadline import DeadlineBudget,DeadlineRequest,compile_launch

UUID = 'GPU-12345678-1234-1234-1234-123456789abc'


def test_binding_checks_the_actual_context_uuid_not_just_environment(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(device,'_binding',None)
    monkeypatch.setattr(device,'_verified',None)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'set_device',lambda n:None)
    monkeypatch.setattr(device,'_current_context_uuid',lambda:UUID)
    proof = device.require_cuda_binding(UUID)
    assert proof['actual_uuid'] == UUID and os.environ['CUDA_VISIBLE_DEVICES'] == UUID
    monkeypatch.setattr(device,'_current_context_uuid',lambda:'GPU-00000000-0000-0000-0000-000000000000')
    with pytest.raises(RuntimeError,match='Actual CUDA'): device.require_cuda_binding(UUID)


def test_prior_cuda_initialization_and_conflicting_visibility_fail(monkeypatch):
    monkeypatch.setattr(device,'_binding',None)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    with pytest.raises(RuntimeError,match='precede'): device.bind_cuda_uuid(UUID)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(RuntimeError,match='conflicts'): device.bind_cuda_uuid(UUID)


def test_deadline_unit_inherits_the_same_single_uuid():
    request = DeadlineRequest(('/usr/bin/python','-m','owned_guard'),'/workspace','a'*32,gpu_uuid=UUID)
    plan = compile_launch(request,DeadlineBudget(100.,120.),now=101.)
    assert '--property=Environment=CUDA_VISIBLE_DEVICES='+UUID in plan.argv


def test_driver_uuid_bytes_are_taken_from_the_current_context(monkeypatch):
    def context(pointer):
        ctypes.cast(pointer,ctypes.POINTER(ctypes.c_int))[0] = 7
        return 0
    def identifier(pointer,current):
        assert current.value == 7
        ctypes.memmove(pointer,uuid.UUID(UUID[4:]).bytes,16)
        return 0
    library = SimpleNamespace(cuCtxGetDevice=context,cuDeviceGetUuid_v2=identifier)
    monkeypatch.setattr(ctypes,'WinDLL' if os.name == 'nt' else 'CDLL',lambda name:library)
    monkeypatch.setattr(torch,'empty',lambda *args,**kwargs:None)
    assert device._current_context_uuid() == UUID
