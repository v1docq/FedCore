"""Versioned, isolated compression runtime; no FEDOT, web, or Dask imports."""
from .contracts import CompressionRequest, InputSpec, DeviceProfile, DataRoles, Resources, WeightedOptions, MethodOptions, ContractError, plan_request

__all__ = ["CompressionRequest", "InputSpec", "DeviceProfile", "DataRoles", "Resources", "WeightedOptions", "MethodOptions", "ContractError", "plan_request"]
CONTRACT_VERSION = 1
SUPPORTED_CONTRACT_VERSIONS = (1, 2, 3)
