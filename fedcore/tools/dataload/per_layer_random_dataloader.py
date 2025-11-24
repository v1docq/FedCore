import torch
from torch.utils.data import TensorDataset
import torch.nn as nn

from torch.utils.data import DataLoader

def generate_batch_per_weight_shape(module: nn.Module,
                               batch_size: int = 8,
                               default_spatial: int = 32,
                               default_seq_len: int = 16,
                               device: str = "cpu",
                               dtype=torch.float32):
    """
    Простая утилита: по форме module.weight генерирует синтетический батч подходящей формы.
    Правила:
      - weight.ndim == 4 -> conv2d: (out, in, kH, kW) -> возвращает (B, in, H, W)
        где H,W = max(k, default_spatial) или k*4 для устойчивости
      - weight.ndim == 3 -> conv1d: (out, in, kL) -> возвращает (B, in, L)
      - weight.ndim == 2 -> linear / embedding weights:
         * если модуль - nn.Embedding -> возвращает LongTensor индексов shape (B, seq_len)
         * иначе -> возвращает (B, in_features)
      - weight.ndim == 1 -> возвращает (B, weight.shape[0])
    """

    if not hasattr(module, "weight") or module.weight is None:
        raise ValueError("Модуль не имеет weight атрибута.")

    w = module.weight
    ndim = w.dim()

    if ndim == 4:
        # conv2d
        in_ch = int(w.shape[1])
        kH, kW = int(w.shape[2]), int(w.shape[3])
        H = max(kH * 4, default_spatial)
        W = max(kW * 4, default_spatial)
        return torch.randn(batch_size, in_ch, H, W, device=device, dtype=dtype)

    if ndim == 3:
        # conv1d weight: (out, in, kL)
        in_ch = int(w.shape[1])
        kL = int(w.shape[2])
        L = max(kL * 4, default_seq_len)
        return torch.randn(batch_size, in_ch, L, device=device, dtype=dtype)

    if ndim == 2:
        # possible: Linear (out, in) or Embedding (num_embeddings, emb_dim)
        if isinstance(module, nn.Embedding) or hasattr(module, "num_embeddings"):
            num = getattr(module, "num_embeddings", None)
            if num is None:
                num = w.shape[0]
            # return indices
            return torch.randint(low=0, high=int(num), size=(batch_size, default_seq_len), dtype=torch.long, device=device)
        else:
            in_f = int(w.shape[1])
            return torch.randn(batch_size, in_f, device=device, dtype=dtype)

    if ndim == 1:
        # bias-like vector
        length = int(w.shape[0])
        return torch.randn(batch_size, length, device=device, dtype=dtype)

    raise NotImplementedError(f"Не поддерживаемая размерность weight: {ndim}")


def create_synthetic_dataloader_per_module_shape(module: nn.Module,
                                            batch_size: int = 8,
                                            n_batches: int = 8,
                                            num_classes: int = 10,
                                            dtype=torch.float32,
                                            default_spatial: int = 32,
                                            default_seq_len: int = 16,
                                            shuffle: bool = False,
                                            device = "cpu") -> DataLoader:
    """
    Создаёт DataLoader из n_batches батчей, сгенерированных по форме module.weight.
    Каждый элемент датасета -> (x, y), где y — случайная метка в [0, num_classes).
    """
    # Собираем батчи
    xs = []
    ys = []
    for _ in range(n_batches):
        x = generate_batch_per_weight_shape(
            module,
            batch_size=batch_size,
            default_spatial=default_spatial,
            default_seq_len=default_seq_len,
            device=device,
            dtype=dtype
        )
        # метки для каждого сэмпла батча
        y = torch.randint(0, num_classes, (x.shape[0],), dtype=torch.long)
        xs.append(x)
        ys.append(y)

    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)

    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)