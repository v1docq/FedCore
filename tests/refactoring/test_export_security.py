import io
import json
import zipfile
import struct
import numpy as np
import pytest
import torch
from fedcore.external_runtime.security import safe_load, safe_save
from fedcore.external_runtime.contracts import ContractError


def archive(path, manifest, arrays=(), compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression=compression) as output:
        output.writestr("manifest.json", json.dumps(manifest))
        for name, data in arrays:
            output.writestr(name, data)


def test_pickle_never_reaches_a_python_deserializer(tmp_path, monkeypatch):
    path = tmp_path / "untrusted.fcb"
    torch.save({"safe_looking": torch.zeros(1)}, path)
    called = []
    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError("Python pickle loader must never be invoked")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ContractError) as error:
        safe_load(path)
    assert error.value.code == "unsafe_format"
    assert not called


def test_huge_npy_shape_is_rejected_before_tensor_allocation(tmp_path, monkeypatch):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "<f4", "fortran_order": False, "shape": (2**40,)})
    path = tmp_path / "huge.fcb"
    manifest = {"kind": "fedcore_tensor_archive", "version": 1, "tree": {"type": "tensor", "name": "t00000.npy"}}
    archive(path, manifest, [("t00000.npy", header.getvalue())])
    called = []
    monkeypatch.setattr(torch, "from_numpy", lambda value: called.append(value))
    with pytest.raises(ContractError) as error:
        safe_load(path)
    assert error.value.code == "invalid_tensor"
    assert not called


@pytest.mark.parametrize("version", [True, "1", 2])
def test_invalid_versions_are_not_coerced(tmp_path, version):
    path = tmp_path / "version.fcb"
    archive(path, {"kind": "fedcore_tensor_archive", "version": version, "tree": {"type": "scalar", "value": 1}})
    with pytest.raises(ContractError):
        safe_load(path)


def test_compression_and_duplicate_members_are_rejected(tmp_path):
    manifest = {"kind": "fedcore_tensor_archive", "version": 1, "tree": {"type": "scalar", "value": 1}}
    compressed = tmp_path / "compressed.fcb"
    archive(compressed, manifest, compression=zipfile.ZIP_DEFLATED)
    with pytest.raises(ContractError):
        safe_load(compressed)
    duplicate = tmp_path / "duplicate.fcb"
    with zipfile.ZipFile(duplicate, "w") as output:
        output.writestr("manifest.json", json.dumps(manifest))
        with pytest.warns(UserWarning):
            output.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(ContractError):
        safe_load(duplicate)


def test_corrupt_crc_has_a_stable_contract_error(tmp_path):
    path = safe_save(torch.zeros(4), tmp_path / "crc.fcb")
    raw = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as source:
        member = source.getinfo("t00000.npy")
        offset = member.header_offset
        filename_size, extra_size = struct.unpack_from("<HH", raw, offset + 26)
        raw[offset+30+filename_size+extra_size+member.file_size-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ContractError) as error:
        safe_load(path)
    assert error.value.code == "invalid_payload"


def test_truncated_zip_is_rejected_with_a_stable_error(tmp_path):
    path = safe_save(torch.zeros(4), tmp_path / "truncated.fcb")
    path.write_bytes(path.read_bytes()[:-15])
    with pytest.raises(ContractError) as error:
        safe_load(path)
    assert error.value.code in ("invalid_payload", "unsafe_format")


def test_nested_round_trip_and_unsupported_object_dtype(tmp_path):
    value = {"x": torch.arange(3), "config": (None, True, 2, ["value", 1.25])}
    restored = safe_load(safe_save(value, tmp_path / "ok.fcb"))
    assert restored["config"] == value["config"]
    torch.testing.assert_close(restored["x"], value["x"])
    bad = io.BytesIO()
    np.save(bad, np.array([{"object": True}], dtype=object), allow_pickle=True)
    path = tmp_path / "object.fcb"
    archive(path, {"kind": "fedcore_tensor_archive", "version": 1, "tree": {"type": "tensor", "name": "t00000.npy"}}, [("t00000.npy", bad.getvalue())])
    with pytest.raises(ContractError):
        safe_load(path)
