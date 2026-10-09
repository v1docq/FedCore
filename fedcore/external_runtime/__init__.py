"""Versioned, isolated compression runtime; no FEDOT, web, or Dask imports."""
from .contracts import CompressionRequest, InputSpec, DeviceProfile, DataRoles, Resources, ContractError, plan_request

__all__ = ["CompressionRequest", "InputSpec", "DeviceProfile", "DataRoles", "Resources", "ContractError", "plan_request"]
CONTRACT_VERSION = 1
