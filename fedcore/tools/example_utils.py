import os

from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler


def get_scenario_for_api(scenario_type: str = 'from_checkpoint', initial_assumption: Union[str, dict] = None):
    if scenario_type.__contains__('checkpoint'):
        learning_strategy = 'from_checkpoint'
    elif scenario_type.__contains__('scratch'):
        learning_strategy = 'from_scratch'
    return initial_assumption, learning_strategy


class CustomForecastingDataset(Dataset):
    def __init__(self, data, well_ids, seq_len, pred_len, target_indices, scaler=None, scaling_factor=1.0,
                 exog_indices=None):
        self.data = data
        self.well_ids = np.unique(well_ids)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_indices = target_indices
        self.exog_indices = exog_indices if exog_indices is not None else target_indices
        self.scaler = scaler
        self.scaling_factor = scaling_factor
        # Find valid starting indices
        self.valid_starts = []
        current_well = well_ids[0]
        consecutive_count = 1

        for i in range(1, len(well_ids)):
            if well_ids[i] == current_well:
                consecutive_count += 1
                if consecutive_count >= seq_len + pred_len:
                    self.valid_starts.append((i - consecutive_count + 1, current_well))
            else:
                current_well = well_ids[i]
                consecutive_count = 1

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx, well_id = self.valid_starts[idx]
        # Get the entire window and scale it as one piece
        full_window = self.data[start_idx:start_idx + self.seq_len + self.pred_len]

        if self.scaler is not None:
            full_window_scaled = self.scaler.transform(full_window)
        else:
            full_window_scaled = full_window

        # Split into historical and future parts after scaling
        x_hist = full_window_scaled[:self.seq_len]
        future_window = full_window_scaled[self.seq_len:]

        x_fut = future_window[:, self.exog_indices]
        y = future_window[:, self.target_indices]

        # Convert to tensors
        x_hist = torch.FloatTensor(x_hist)
        x_fut = torch.FloatTensor(x_fut)
        y = torch.FloatTensor(y)
        x_fut[..., :] = x_fut[..., :] * self.scaling_factor
        return x_hist, x_fut, y


def get_custom_dataloader(task_params):
    dataset_names = os.listdir(task_params['path_to_dataset'])
    dataset_list = [pd.read_csv(os.path.join(task_params['path_to_dataset'], name)) for name in dataset_names]
    exog_indices = [task_params['feature_columns'].index(col) for col in task_params['exog_columns']]
    dataset_list = [df[task_params['feature_columns'] + ['WELL_ID']] for df in dataset_list]
    dataset_names = [x.split('.')[0] for x in dataset_names]
    # Map names to indices
    target_indices = [0]  # Keep track of names in order
    scaler = MinMaxScaler()
    train_data = dataset_list[0][task_params['feature_columns']].values
    shuffle = [True, True, False]
    scaler.fit(train_data)
    ts_dataset_list = [CustomForecastingDataset(data=df[task_params['feature_columns']].values,
                                                well_ids=df['WELL_ID'].values,
                                                target_indices=target_indices,
                                                exog_indices=exog_indices,
                                                seq_len=task_params['seq_len'][df_name],
                                                pred_len=task_params['pred_len'][df_name],
                                                scaler=scaler,
                                                scaling_factor=task_params['scaling_factor'][df_name],
                                                ) for df, df_name in zip(dataset_list, dataset_names)]
    torch_loader_dict = {f'{name}_dataloader': DataLoader(dataset,
                                                          batch_size=task_params['batch_size'][name],
                                                          shuffle=is_shuffled,
                                                          num_workers=8,  # Adjust based on CPU cores
                                                          pin_memory=True  # Faster data transfer to GPU
                                                          ) for dataset, name, is_shuffled in
                         zip(ts_dataset_list, dataset_names, shuffle)}
    return torch_loader_dict
