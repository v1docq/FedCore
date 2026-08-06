"""Build a JSON-serializable hierarchical graph from a PyTorch module.

I/O contract
------------
Input:  ``torch.nn.Module`` only.
Output: ``ModelGraphView`` — module hierarchy + leaf list for the Web UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch.nn as nn


@dataclass
class GraphNode:
    id: str
    name: str
    type: str
    param_count: int
    is_leaf: bool
    depth: int
    parent: Optional[str] = None
    index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    kind: str = "contains"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelGraphView:
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    modules: List[Dict[str, Any]] = field(default_factory=list)
    total_modules: int = 0
    model_class: str = ""
    layout: str = "hierarchy"

    def to_dict(self) -> Dict[str, Any]:
        # ``layers`` kept as alias of ``modules`` for older UI code paths
        return {
            "model_class": self.model_class,
            "layout": self.layout,
            "total_modules": self.total_modules,
            "total_layers": self.total_modules,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "modules": list(self.modules),
            "layers": list(self.modules),
        }


class ModelGraphBuilder:
    """Hierarchy from ``named_modules`` (containers + leaves)."""

    def build(self, model: nn.Module) -> ModelGraphView:
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Expected torch.nn.Module, got {type(model).__name__}. "
                "Upload a full module checkpoint, not a bare state_dict."
            )

        name_to_id: Dict[str, str] = {"": "root"}
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        modules: List[Dict[str, Any]] = []

        root_params = sum(p.numel() for p in model.parameters())
        nodes.append(
            GraphNode(
                id="root",
                name=type(model).__name__,
                type=type(model).__name__,
                param_count=root_params,
                is_leaf=not any(True for _ in model.children()),
                depth=0,
                parent=None,
            )
        )

        leaf_index = 0
        for qual_name, module in model.named_modules():
            if qual_name == "":
                continue

            node_id = self._id_for(qual_name)
            name_to_id[qual_name] = node_id
            parent_qual = ".".join(qual_name.split(".")[:-1])
            parent_id = name_to_id.get(parent_qual, "root")
            depth = qual_name.count(".") + 1
            is_leaf = not any(True for _ in module.children())
            if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
                is_leaf = False

            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            if not is_leaf:
                param_count = sum(p.numel() for p in module.parameters())

            short_name = qual_name.split(".")[-1]
            nodes.append(
                GraphNode(
                    id=node_id,
                    name=qual_name,
                    type=type(module).__name__,
                    param_count=param_count,
                    is_leaf=is_leaf,
                    depth=depth,
                    parent=parent_id,
                    index=leaf_index if is_leaf else None,
                )
            )
            edges.append(GraphEdge(source=parent_id, target=node_id, kind="contains"))

            if is_leaf:
                modules.append(
                    {
                        "id": node_id,
                        "index": leaf_index,
                        "name": qual_name,
                        "short_name": short_name,
                        "type": type(module).__name__,
                        "param_count": sum(
                            p.numel() for p in module.parameters(recurse=False)
                        ),
                        "depth": depth,
                        "parent": parent_id,
                        "group": qual_name.split(".")[0],
                    }
                )
                leaf_index += 1

        return ModelGraphView(
            nodes=nodes,
            edges=edges,
            modules=modules,
            total_modules=len(modules),
            model_class=type(model).__name__,
            layout="hierarchy",
        )

    @staticmethod
    def _id_for(qual_name: str) -> str:
        safe = qual_name.replace(".", "__").replace("-", "_")
        return f"m_{safe}" if safe else "root"


def build_model_graph(model: nn.Module) -> Dict[str, Any]:
    return ModelGraphBuilder().build(model).to_dict()
