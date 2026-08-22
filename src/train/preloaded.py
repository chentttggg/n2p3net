"""预上传训练集 DataLoader（GTN-N2P3Net 大 fold 性能项）。

Trainer 消费标准 DataLoader 接口；标准 DataLoader 每 epoch 都会从 CPU 重复搬运 batch。
GTN LOSO 每 fold 约 38k trials，且要训 20–50 epochs，重复 H2D 是主要浪费之一。
本模块把 X/y/domain_id 一次性上传到训练设备，之后每 epoch 只在设备端做 randperm +
slice，避免重复 H2D（与 deep 基线 D-deep-upload 同一策略）。
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch


class PreloadedDataLoader:
    """已上传设备的 (X, y[, domain_id]) 批量迭代器，接口对齐 DataLoader 的 batch 产出。"""

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        domain_id: Optional[torch.Tensor] = None,
        *,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        device: Optional[torch.device] = None,
    ):
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y 长度不一致：{X.shape[0]} vs {y.shape[0]}。")
        if domain_id is not None and domain_id.shape[0] != X.shape[0]:
            raise ValueError(f"X/domain_id 长度不一致：{X.shape[0]} vs {domain_id.shape[0]}。")
        if batch_size <= 0:
            raise ValueError(f"batch_size 须 >0，得到 {batch_size}。")

        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        if device is not None:
            self.X = X.to(device)
            self.y = y.to(device, dtype=torch.float32)
            self.domain_id = domain_id.to(device, dtype=torch.long) if domain_id is not None else None
        else:
            self.X = X
            self.y = y
            self.domain_id = domain_id

        self.n = self.X.shape[0]
        self._epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        if self.shuffle:
            gen = torch.Generator()
            gen.manual_seed(self.seed + self._epoch)
            self._epoch += 1
            perm = torch.randperm(self.n, generator=gen)
        else:
            perm = torch.arange(self.n)

        for start in range(0, self.n, self.batch_size):
            idx = perm[start : start + self.batch_size]
            if self.drop_last and idx.shape[0] < self.batch_size:
                break
            if self.domain_id is None:
                yield self.X[idx], self.y[idx]
            else:
                yield self.X[idx], self.y[idx], self.domain_id[idx]
