"""Non-pickle tensor archives: bounded JSON plus strictly bounded raw NPY arrays.

No torch.load or pickle implementation is involved, including on older torch.
Archive members are read in memory and never extracted to filesystem paths.
"""
from __future__ import annotations
import io
import json
import math
import re
import zipfile
from pathlib import Path
import numpy as np
import torch
from .contracts import ContractError, relative_name

ARCHIVE_KIND = "fedcore_tensor_archive"
_DTYPES = {"float32", "float64", "int64", "int32", "bool"}


def confined_path(root, name, *, must_exist=True):
    root = Path(root).resolve()
    path = (root / relative_name(name)).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ContractError("invalid_path", "Artifact escaped the job directory", "path")
    if must_exist and not path.is_file():
        raise ContractError("missing_artifact", f"Missing artifact: {name}", "path")
    return path


def safe_save(value, path):
    """Write the explicitly named .fcb archive format; never masquerade as .pt."""
    path = Path(path)
    if path.suffix != ".fcb":
        raise ContractError("unsafe_format", "Tensor archives require the .fcb suffix")
    arrays = {}
    count = [0]
    def encode(item, depth=0):
        count[0] += 1
        if depth > 16 or count[0] > 10000:
            raise ContractError("invalid_payload", "Archive nesting or node count exceeds limits")
        if type(item) is torch.Tensor:
            if item.layout != torch.strided or str(item.dtype).removeprefix("torch.") not in _DTYPES:
                raise ContractError("invalid_tensor", "Unsupported tensor layout or dtype")
            name = f"t{len(arrays):05d}.npy"
            buffer = io.BytesIO()
            np.save(buffer, item.detach().cpu().contiguous().numpy(), allow_pickle=False)
            arrays[name] = buffer.getvalue()
            return {"type": "tensor", "name": name}
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise ContractError("invalid_payload", "Archive keys must be strings")
            return {"type": "dict", "items": [[k, encode(v, depth+1)] for k, v in item.items()]}
        if type(item) in (tuple, list):
            return {"type": "tuple" if type(item) is tuple else "list", "items": [encode(v, depth+1) for v in item]}
        if item is None or type(item) in (str, bool, int, float):
            return {"type": "scalar", "value": item}
        raise ContractError("unsafe_payload", "Unknown Python objects cannot be serialized")
    manifest = {"kind": ARCHIVE_KIND, "version": 1, "tree": encode(value)}
    raw = json.dumps(manifest, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(raw) > 65536:
        raise ContractError("size_limit", "Archive descriptor exceeds 64 KiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", raw)
        for name, data in arrays.items():
            archive.writestr(name, data)
    return path


def _tensor_from_npy(raw, max_bytes):
    try:
        buffer = io.BytesIO(raw)
        version = np.lib.format.read_magic(buffer)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(buffer, max_header_size=16384)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(buffer, max_header_size=16384)
        else:
            raise ValueError("Unsupported NPY version")
        if not isinstance(shape, tuple) or len(shape) > 8 or any(type(n) is not int or not 0 <= n <= 100000000 for n in shape):
            raise ValueError("Invalid tensor dimensions")
        size = math.prod(shape) * dtype.itemsize
        if size > max_bytes or dtype.name not in _DTYPES or dtype.hasobject or dtype.byteorder not in ("=", "<", "|") or fortran:
            raise ValueError("Unsupported tensor storage")
        if len(raw) != buffer.tell() + size:
            raise ValueError("Declared tensor shape does not match raw byte length")
        array = np.frombuffer(raw, dtype=dtype, count=math.prod(shape), offset=buffer.tell()).reshape(shape).copy()
        tensor = torch.from_numpy(array)
        if not torch.isfinite(tensor).all():
            raise ValueError("Tensor values must be finite")
        return tensor
    except Exception as error:
        raise ContractError("invalid_tensor", "Rejected bounded NPY tensor header or storage") from error


def _read_archive(path, max_bytes):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ContractError("size_limit", "Artifact is absent or exceeds the byte limit")
    if path.suffix != ".fcb" or not zipfile.is_zipfile(path):
        raise ContractError("unsafe_format", "Only non-pickle .fcb tensor archives are accepted")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if len(names) != len(set(names)) or "manifest.json" not in names or any(n != "manifest.json" and not re.fullmatch(r"t[0-9]{5}\.npy", n) for n in names):
            raise ContractError("unsafe_format", "Unrecognized archive layout; pickle archives are forbidden")
        if len(members) > 10001 or sum(m.file_size for m in members) > max_bytes or any(m.flag_bits & 1 or m.compress_type != zipfile.ZIP_STORED for m in members):
            raise ContractError("size_limit", "Expanded archive exceeds the byte/member limit or is encrypted")
        if archive.getinfo("manifest.json").file_size > 65536:
            raise ContractError("size_limit", "Archive descriptor exceeds 64 KiB")
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ContractError("invalid_payload", "Malformed archive JSON") from error
        if not isinstance(manifest, dict) or set(manifest) != {"kind", "version", "tree"} or manifest["kind"] != ARCHIVE_KIND or type(manifest["version"]) is not int or manifest["version"] != 1:
            raise ContractError("unsafe_format", "Unknown tensor archive version")
        used, count = set(), [0]
        def decode(node, depth=0):
            count[0] += 1
            if not isinstance(node, dict) or depth > 16 or count[0] > 10000:
                raise ContractError("invalid_payload", "Archive nesting or node count exceeds limits")
            kind = node.get("type")
            if kind == "tensor" and set(node) == {"type", "name"}:
                name = node["name"]
                if not isinstance(name, str) or name not in names or name == "manifest.json" or name in used:
                    raise ContractError("invalid_tensor", "Invalid or duplicated tensor reference")
                used.add(name)
                return _tensor_from_npy(archive.read(name), max_bytes)
            if kind in ("list", "tuple", "dict") and set(node) == {"type", "items"} and isinstance(node["items"], list):
                items = node["items"]
                if kind == "dict":
                    if any(not isinstance(i, list) or len(i) != 2 or not isinstance(i[0], str) for i in items) or len({i[0] for i in items}) != len(items):
                        raise ContractError("invalid_payload", "Invalid or duplicate object keys")
                    return {k: decode(v, depth+1) for k, v in items}
                values = [decode(v, depth+1) for v in items]
                return tuple(values) if kind == "tuple" else values
            if kind == "scalar" and set(node) == {"type", "value"}:
                value = node["value"]
                if value is None or type(value) in (str, bool, int) or type(value) is float and math.isfinite(value):
                    return value
            raise ContractError("invalid_payload", "Unknown or malformed archive tree node")
        value = decode(manifest["tree"])
        if used != set(names) - {"manifest.json"}:
            raise ContractError("invalid_payload", "Archive contains unreferenced tensors")
        return value


def safe_load(path, max_bytes=64 * 1024 * 1024):
    try:
        return _read_archive(path, max_bytes)
    except ContractError:
        raise
    except (zipfile.BadZipFile, RuntimeError, ValueError, EOFError, OverflowError) as error:
        raise ContractError("invalid_payload", "Corrupt or unsupported tensor archive") from error
