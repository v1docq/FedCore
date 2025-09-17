from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_benchmark_data(dataset_id: int = 110):
    # fetch dataset
    dataset = fetch_ucirepo(id=dataset_id)

    return dict(features=dataset.data.features,
                target=dataset.data.targets,
                metadata=dataset.metadata)


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
