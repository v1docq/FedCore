"""Optional modern FEDOT discovery module. Importing it never registers models."""
from .adapters import build_industrial_extension_manifest

FEDOT_EXTENSION_MANIFEST = build_industrial_extension_manifest()
