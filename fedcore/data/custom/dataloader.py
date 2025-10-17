from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from fedcore.tools.logging import setup_logging


@dataclass
class DatasetMetadata:
    name: str
    source: str
    shape: tuple
    processed_shape: Optional[tuple] = None
    feature_columns: Optional[List[str]] = None
    label_columns: Optional[List[str]] = None


class DatasetLoader:
    def __init__(self, dataset_config: Dict[str, Any]):
        self.dataset_config = dataset_config or {}
        self.logger = setup_logging()

    def _load(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        config = self.dataset_config[dataset_name]
        result = None

        match config['source']:
            case 'openml':
                result = self._load_openml_dataset(dataset_name, config)
            case 'uci':
                result = self._load_uci_dataset(dataset_name, config)
            case 'custom':
                result = config['method'](**config['method_params'])
            case _:
                self.logger.error(f"Unknown dataset source: {config['source']} for dataset {dataset_name}")

        return result

    def _load_openml_dataset(self, name: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            self.logger.info(f"Fetching OpenML dataset ID: {config['id']}")
            dataset = fetch_openml(config['id'], version=config.get('version', 'active'), as_frame=True)
            self.logger.info(f"Successfully loaded OpenML dataset: {name} with shape {dataset.data.shape}")
            return {
                'features': dataset.data,
                'target': dataset.target,
                'metadata': DatasetMetadata(
                    name=name,
                    source=config['source'],
                    shape=dataset.data.shape,
                )
            }
        except Exception as e:
            self.logger.error(f"Failed to load OpenML dataset {name}: {str(e)}")
            return None

    def _load_uci_dataset(self, name: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            self.logger.info(f"Fetching UCI dataset ID: {config['id']}")
            dataset = fetch_ucirepo(id=config['id'])
            self.logger.info(f"Successfully loaded UCI dataset: {name} with shape {dataset.data.features.shape}")
            return {
                'features': dataset.data.features,
                'target': dataset.data.targets,
                'metadata': DatasetMetadata(
                    name=name,
                    source=config['source'],
                    shape=dataset.data.features.shape,
                )
            }
        except Exception as e:
            self.logger.error(f"Failed to load UCI dataset {name}: {str(e)}")
            return None

    def load(self) -> Dict[str, Any]:
        results = {}

        for name in self.dataset_config.keys():
            results[name] = self._load(name)
        return results
