import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesForecastingDataset(Dataset):
    def __init__(self,
                 time_series,
                 history_len,
                 forecast_len,
                 target_indices=None,
                 well_ids=None,
                 exog_indices=None):
        # Define endog params of ts. If this a case of 1-d forecasting here we stop
        self.data = time_series
        self.history_len = history_len
        self.forecast_len = forecast_len

        # Define exog params. If this a case of multidim forecasting we looking for exog variables and some metadata

    def _exog_preproc(self):
        self.well_ids = np.unique(well_ids)
        self.target_indices = target_indices
        self.exog_indices = exog_indices if exog_indices is not None else target_indices

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

        # Split into historical and future parts after scaling
        x_hist = full_window[:self.seq_len]
        future_window = full_window[self.seq_len:]

        x_fut = future_window[:, self.exog_indices]
        y = future_window[:, self.target_indices]

        # Convert to tensors
        x_hist = torch.FloatTensor(x_hist)
        x_fut = torch.FloatTensor(x_fut)
        y = torch.FloatTensor(y)
        x_fut[..., :] = x_fut[..., :]
        return x_hist.T, x_fut.T, y.T
