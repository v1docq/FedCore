"""Explicit support boundary for audited compression modes.

This catalog describes modes, not model/backend applicability. An operation
still has to validate its inputs before it changes a model.
"""
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from importlib import resources
import json
from types import MappingProxyType
from typing import Mapping


class SupportLevel(str, Enum):
    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ModeCapability:
    name: str
    support: SupportLevel
    contract: str
    checks: tuple[str, ...]
    aliases: tuple[str, ...] = ()


class UnsupportedModeError(ValueError):
    """An unavailable mode was requested before execution."""
    def __init__(self, mode: str, support: SupportLevel, reason: str):
        self.code = "experimental_mode" if support is SupportLevel.EXPERIMENTAL else "unsupported_mode"
        self.mode = mode
        self.support = support
        self.reason = reason
        super().__init__(f"{mode}: {reason} ({support.value})")


def parse_catalog(payload) -> tuple[ModeCapability, ...]:
    if not isinstance(payload, dict) or type(payload.get("version")) is not int or payload.get("version") != 1:
        raise ValueError("Unsupported capability catalog version")
    entries = payload.get("modes")
    if not isinstance(entries, list):
        raise ValueError("Capability modes must be a list")
    result, names = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"name", "support", "contract", "checks", "aliases"}:
            raise ValueError("Invalid capability entry")
        name, contract = entry.get("name"), entry.get("contract")
        checks, aliases = entry.get("checks"), entry.get("aliases", [])
        if not isinstance(name, str) or not name or not isinstance(contract, str) or not contract:
            raise ValueError("Capability name and contract must be nonempty strings")
        if not isinstance(checks, list) or not checks or not all(isinstance(c, str) and c for c in checks):
            raise ValueError("A capability requires named checks")
        if not isinstance(aliases, list) or not all(isinstance(a, str) and a for a in aliases):
            raise ValueError("Invalid capability aliases")
        keys = [name, *aliases]
        if len(set(keys)) != len(keys) or any(key in names for key in keys):
            raise ValueError("Duplicate capability name or alias")
        names.update(keys)
        result.append(ModeCapability(name, SupportLevel(entry.get("support")), contract, tuple(checks), tuple(aliases)))
    return tuple(result)


@lru_cache(maxsize=1)
def capabilities() -> tuple[ModeCapability, ...]:
    source = resources.files("fedcore.repository").joinpath("data", "mode_capabilities.json")
    return parse_catalog(json.loads(source.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def _index():
    return MappingProxyType({key: item for item in capabilities() for key in (item.name, *item.aliases)})


def capability_for(mode: str) -> ModeCapability:
    try:
        return _index()[mode]
    except (KeyError, TypeError):
        raise UnsupportedModeError(str(mode), SupportLevel.UNSUPPORTED, "Unknown compression mode") from None


def require_supported(mode: str) -> ModeCapability:
    item = capability_for(mode)
    if item.support is not SupportLevel.SUPPORTED:
        raise UnsupportedModeError(item.name, item.support, item.contract)
    return item


def validate_requested_modes(parameters) -> None:
    """Reject audited experimental aliases in configuration before effects.

    A quantization type called ``dynamic`` is intentionally not interpreted
    as the unrelated experimental dynamic-rank pruner.
    """
    if not isinstance(parameters, Mapping) and not hasattr(parameters, "items"):
        return
    guarded = {"importance", "rank_pruner", "rank_pruning_strategy", "lr_hook", "hooks"}
    blocked = {"cuttlefish", "DynamicRankPruner", "activation_entropy", "custom_depth", "ImportancePreservingRankSelector", "manifold_losses"}
    for key, value in parameters.items():
        if isinstance(value, Mapping) or hasattr(value, "items"):
            validate_requested_modes(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping) or hasattr(item, "items"):
                    validate_requested_modes(item)
                elif key in guarded:
                    name = _mode_name(item)
                    if name in blocked:
                        require_supported(name)
        elif key in guarded:
            name = _mode_name(value)
            if name in blocked:
                require_supported(name)


def _mode_name(value):
    if isinstance(value, str):
        return value
    if isinstance(value, Enum):
        return value.name
    return getattr(value, "__name__", None)
