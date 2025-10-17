import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from typing import Dict, Any

from fedcore.data.custom.dataloader import DatasetLoader
from fedcore.tools.logging import setup_logging


# TODO: remove/refactor legacy method
def load_benchmark_data(dataset_id: int = 110):
    # fetch dataset
    dataset = fetch_ucirepo(id=dataset_id)

    return dict(features=dataset.data.features,
                target=dataset.data.targets,
                metadata=dataset.metadata)

# TODO: remove/refactor legacy method
def split_benchmark_data(dataset_dict: dict, use_subsample=None):
    X = dataset_dict['features'].values.astype('float32')
    y = dataset_dict['target'].values
    if y.dtype == object:
        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    return dict(train_features=X,
                test_features=X_test,
                train_target=y,
                test_target=y_test)


def preprocess_dataset(X, dataset_name: str):
    logger = setup_logging()
    logger.info(f"Starting preprocessing for dataset: {dataset_name}")

    if isinstance(X, pd.Series):
        X = X.to_frame()
        was_series = True
    else:
        was_series = False

    logger.info(f"Initial shape: {X.shape}, data types: {X.dtypes.value_counts().to_dict()}")

    X_processed = X.copy()
    nan_count = X_processed.isna().sum().sum()
    
    if nan_count > 0:
        logger.warning(f"Dataset {dataset_name} contains {nan_count} NaN values")
        initial_shape = X_processed.shape
        X_processed = X_processed.dropna(thresh=X_processed.shape[1] // 2)
        rows_dropped = initial_shape[0] - X_processed.shape[0]
        if rows_dropped > 0:
            logger.info(f"Dropped {rows_dropped} rows with excessive NaN values in {dataset_name}")

        for column in X_processed.columns:
            if X_processed[column].isna().any():
                column_nan_count = X_processed[column].isna().sum()
                logger.info(f"Processing column {column} with {column_nan_count} NaN values in {dataset_name}")
                if X_processed[column].dtype == 'object':
                    fill_value = (X_processed[column].mode()[0] 
                                if not X_processed[column].mode().empty 
                                else 'missing')
                    X_processed[column].fillna(fill_value, inplace=True)
                    logger.debug(f"Filled {column_nan_count} NaN values in categorical column {column} with mode: {fill_value}")
                else:
                    fill_value = X_processed[column].median()
                    X_processed[column].fillna(fill_value, inplace=True)
                    logger.debug(f"Filled {column_nan_count} NaN values in numerical column {column} with median: {fill_value}")
    else:
        logger.info(f"No NaN values found in dataset {dataset_name}")

    categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
    categorical_count = len(categorical_cols)
    
    if categorical_count > 0:
        logger.info(f"Encoding {categorical_count} categorical columns in {dataset_name}")
        encoders = {}
        for col in categorical_cols:
            unique_count = X_processed[col].nunique()
            logger.info(f"Encoding categorical column {col} with {unique_count} unique values")
            encoder = LabelEncoder()
            X_processed[col] = encoder.fit_transform(X_processed[col].astype(str))
            encoders[col] = encoder
            logger.debug(f"Completed encoding for column {col}")
    else:
        logger.info(f"No categorical columns found in dataset {dataset_name}")

    if was_series:
        X_processed = X_processed.iloc[:, 0]

    return X_processed


def load_and_preprocess_datasets(dataset_config: Dict[str, Any]) -> Dict[str, Any]:
    loader = DatasetLoader(dataset_config)
    datasets = loader.load()
    processed_datasets = {}

    for name, dataset_info in datasets.items():
        features = dataset_info['features']
        target = dataset_info['target']
        metadata = dataset_info['metadata']

        processed_features = preprocess_dataset(features, name)
        processed_target = preprocess_dataset(target, f"{name}_target")
        
        metadata.processed_shape = processed_features.shape

        processed_datasets[name] = {
            'features': processed_features,
            'target': processed_target,
            'metadata': metadata,
        }
    return processed_datasets


if __name__ == "__main__":
    dataset_config = {
        'genbase': {'id': 'genbase', 'source': 'openml', 'version': 2},
        'yeast': {'id': 'yeast', 'source': 'openml', 'version': 1},
        'birds': {'id': 'birds', 'source': 'openml', 'version': 3},
        'rt_iot2022': {'id': 942, 'source': 'uci'},
        # 'mediamill': {'id': 'mediamill', 'source': 'direct',
        #               'method': download_specific_mediamill,
        #               'method_params': {'experiment': 'exp1'}},
    }
    datasets = load_and_preprocess_datasets(dataset_config)
