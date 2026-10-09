# Measured FedCore examples

Run from the repository root with the same Python environment as FedCore:

```text
python -m examples.petra.run --scenario tabular --epochs 20 --finetune-epochs 3 --methods baseline train svd --output-dir work/petra_results
python -m examples.petra.view_results work/petra_results
```

`cv` defaults to the separate real 8x8 digits pilot. CIFAR10/ImageNette require local original data and explicit `--cv-dataset`, `--data`, and architecture options. There are no automatic dataset downloads. A positive baseline training budget is mandatory; unknown historical weights are never used.

The [scenario guide](../docs/petra/examples.md) describes the data, access requirements, roles, units, and evidence limits. [inventory.json](petra/inventory.json) accounts for all 75 original audited files, including consolidated duplicates. Historical logs, figures, and weights are historical artifacts with unverified provenance.

Notebooks are thin readers of verified run records. Stored notebook output is cleared. Original filenames do not establish the actual model architecture, dataset, or task.

The educational storage service in `examples/app.py` does not convert models: `/export` returns an explicit unsupported response. Use the real FedCore export path or the shared runner for an executable measured artifact.

Rebuild the published ZIP from these source files with `python -m examples.petra.package`, then run `python scripts/check_example_secrets.py examples`. The ZIP has a source checksum manifest and contains no measured article results.
