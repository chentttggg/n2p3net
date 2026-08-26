"""预上传训练集 DataLoader（GTN-N2P3Net 大 fold 性能项）。

Trainer 消费标准 DataLoader 接口；标准 DataLoader 每 epoch 都会从 CPU 重复搬运 batch。
GTN LOSO 每 fold 约 38k trials，且要训 20–50 epochs，重复 H2D 是主要浪费之一。
本模块把 X/y/domain_id 一次性上传到训练设备，之后每 epoch 只在设备端做 randperm +
slice，避免重复 H2D（与 deep 基线 D-deep-upload 同一策略）。

性能教训（doc/performance_lessons.md）：上传时一次性完成 finite 校验并标记
``prevalidated``；GTNSetDataLoader 的 repetition/sequence ranks 在构造时生成，
epoch 迭代不再逐 digit 重建。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from train.contracts import SetMetadata, TrialContext


class PreloadedDataLoader:
    """已上传设备的 (X, y[, domain_id]) 批量迭代器，接口对齐 DataLoader 的 batch 产出。"""

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        domain_id: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
        *,
        batch_size: int = 256,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        device: torch.device | None = None,
        validate_finite: bool = True,
    ):
        if X.dim() != 3:
            raise ValueError(f"X must be (N,C,T), got {tuple(X.shape)}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y 长度不一致：{X.shape[0]} vs {y.shape[0]}。")
        labels = y.reshape(-1)
        if labels.numel() != X.shape[0] or not bool(
            torch.isfinite(labels).all() & ((labels == 0) | (labels == 1)).all()
        ):
            raise ValueError("y must contain one finite binary label per trial.")
        if domain_id is not None:
            if domain_id.shape != (X.shape[0],):
                raise ValueError(
                    f"X/domain_id 长度不一致：{X.shape[0]} vs {domain_id.shape[0]}。"
                )
            if domain_id.dtype == torch.bool or domain_id.is_floating_point():
                raise ValueError("domain_id must have an integer dtype.")
        if channel_mask is not None:
            if channel_mask.dtype != torch.bool:
                raise ValueError("channel_mask must have boolean dtype.")
            if channel_mask.shape not in {(X.shape[1],), X.shape[:2]}:
                raise ValueError("channel_mask must be (C,) or (N,C).")
        if batch_size <= 0:
            raise ValueError(f"batch_size 须 >0，得到 {batch_size}。")

        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        if device is not None:
            self.X = X.to(device)
            self.y = y.to(device, dtype=torch.float32)
            self.domain_id = (
                domain_id.to(device, dtype=torch.long) if domain_id is not None else None
            )
            self.channel_mask = (
                channel_mask.to(device, dtype=torch.bool) if channel_mask is not None else None
            )
        else:
            self.X = X
            self.y = y
            self.domain_id = domain_id
            self.channel_mask = channel_mask

        self.n = self.X.shape[0]
        self._epoch = 0
        self.finite_validated = False
        if validate_finite:
            if not bool(torch.isfinite(self.X).all()):
                raise ValueError("Preloaded training samples must be finite.")
            self.finite_validated = True

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    @property
    def full_context(self) -> TrialContext:
        context = TrialContext(
            X=self.X,
            y=self.y,
            domain_id=self.domain_id,
            channel_mask=self.channel_mask,
            prevalidated=self.finite_validated,
        )
        context.validate()
        return context

    def __iter__(self) -> Iterator[TrialContext]:
        random_device = self.X.device if self.X.device.type == "cuda" else torch.device("cpu")
        index_device = self.X.device if self.X.device.type == "cuda" else torch.device("cpu")
        generator = torch.Generator(device=random_device)
        if self.shuffle:
            generator.manual_seed(self.seed + self._epoch)
            self._epoch += 1
            perm = torch.randperm(self.n, generator=generator, device=index_device)
        else:
            perm = torch.arange(self.n, device=index_device)

        for start in range(0, self.n, self.batch_size):
            idx = perm[start : start + self.batch_size]
            if self.drop_last and idx.shape[0] < self.batch_size:
                break
            yield TrialContext(
                X=self.X[idx],
                y=self.y[idx],
                domain_id=self.domain_id[idx] if self.domain_id is not None else None,
                channel_mask=(
                    self.channel_mask[idx]
                    if self.channel_mask is not None and self.channel_mask.dim() == 2
                    else self.channel_mask
                ),
                prevalidated=self.finite_validated,
            )


class GTNSetDataLoader:
    """Emit ragged, chronological GTN prefixes with per-K coverage.

    A K checkpoint is the first acquisition time at which every candidate has
    appeared at least K times.  All valid flashes before that checkpoint remain
    in the sequence; no already-observed evidence is silently discarded.

    ``SetMetadata.prevalidated_kmax`` is the smallest K covered by every GTN
    group in the emitted batch (not the loader-wide maximum K).  v12 objectives
    may only skip per-K emptiness checks up to that guarantee; partial-coverage
    groups still get safe ``has_k`` filtering for larger K.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        stimulus_digits: torch.Tensor,
        set_group_ids: torch.Tensor,
        domain_id: torch.Tensor | None = None,
        acquisition_indices: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
        *,
        evidence_k: int | None = None,
        evidence_ks: Sequence[int] = (1, 3, 5, 10, 15),
        digit_vocab: Sequence[int] = tuple(range(1, 10)),
        batch_size: int = 256,
        shuffle: bool = True,
        seed: int = 0,
        main_domain: int = 0,
        device: torch.device | None = None,
        validate_finite: bool = True,
    ):
        if X.dim() != 3:
            raise ValueError(f"X must be (N,C,T), got {tuple(X.shape)}.")
        n = X.shape[0]
        if not (y.shape[0] == stimulus_digits.shape[0] == set_group_ids.shape[0] == n):
            raise ValueError("X/y/stimulus_digits/set_group_ids lengths must match.")
        if domain_id is not None and domain_id.shape[0] != n:
            raise ValueError("X/domain_id lengths must match.")
        labels = y.reshape(-1)
        if labels.numel() != n or not bool(
            torch.isfinite(labels).all() & ((labels == 0) | (labels == 1)).all()
        ):
            raise ValueError("y must contain one finite binary label per trial.")
        for name, values in (
            ("stimulus_digits", stimulus_digits),
            ("set_group_ids", set_group_ids),
            ("domain_id", domain_id),
            ("acquisition_indices", acquisition_indices),
        ):
            if values is not None and (values.dtype == torch.bool or values.is_floating_point()):
                raise ValueError(f"{name} must have an integer dtype.")
        if acquisition_indices is not None and acquisition_indices.shape != (n,):
            raise ValueError("acquisition_indices must contain one value per trial.")
        if channel_mask is not None:
            if channel_mask.dtype != torch.bool:
                raise ValueError("channel_mask must have boolean dtype.")
            if channel_mask.shape not in {(X.shape[1],), (n, X.shape[1])}:
                raise ValueError("channel_mask must be (C,) or (N,C).")
        if evidence_k is not None:
            evidence_ks = (int(evidence_k),)
        normalized_ks = tuple(sorted({int(k) for k in evidence_ks}))
        if not normalized_ks or normalized_ks[0] < 1:
            raise ValueError("evidence_ks must contain positive integers.")
        max_evidence_k = normalized_ks[-1]
        if batch_size < max_evidence_k * len(digit_vocab):
            raise ValueError(
                "batch_size must fit at least one complete GTN set; got "
                f"{batch_size} < {max_evidence_k * len(digit_vocab)}."
            )

        self.batch_size = int(batch_size)
        self.evidence_ks = normalized_ks
        self.evidence_k = max_evidence_k
        self.digit_vocab = tuple(int(v) for v in digit_vocab)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.main_domain = int(main_domain)
        self._epoch = 0

        if device is not None:
            self.X = X.to(device)
            self.y = y.to(device, dtype=torch.float32)
            self.stimulus_digits = stimulus_digits.to(device, dtype=torch.long)
            self.set_group_ids = set_group_ids.to(device, dtype=torch.long)
            self.domain_id = (
                domain_id.to(device, dtype=torch.long) if domain_id is not None else None
            )
            self.acquisition_indices = (
                acquisition_indices.to(device, dtype=torch.long)
                if acquisition_indices is not None
                else torch.arange(n, device=device, dtype=torch.long)
            )
            self.channel_mask = (
                channel_mask.to(device, dtype=torch.bool) if channel_mask is not None else None
            )
        else:
            self.X = X
            self.y = y.to(dtype=torch.float32)
            self.stimulus_digits = stimulus_digits.to(dtype=torch.long)
            self.set_group_ids = set_group_ids.to(dtype=torch.long)
            self.domain_id = domain_id.to(dtype=torch.long) if domain_id is not None else None
            self.acquisition_indices = (
                acquisition_indices.to(dtype=torch.long)
                if acquisition_indices is not None
                else torch.arange(n, dtype=torch.long)
            )
            self.channel_mask = channel_mask.to(dtype=torch.bool) if channel_mask is not None else None

        self.finite_validated = False
        if validate_finite:
            if not bool(torch.isfinite(self.X).all()):
                raise ValueError("GTN set training samples must be finite.")
            self.finite_validated = True

        main_mask = self.set_group_ids >= 0
        if self.domain_id is not None:
            main_mask &= self.domain_id == self.main_domain
        group_values = torch.unique(self.set_group_ids[main_mask], sorted=True).tolist()
        self._sequences: dict[int, torch.Tensor] = {}
        self._repetition_ranks: dict[int, torch.Tensor] = {}
        self._sequence_ranks: dict[int, torch.Tensor] = {}
        self._largest_available_k: dict[int, int] = {}
        all_counts: dict[int, list[int]] = {}
        for group in group_values:
            rows = torch.nonzero(
                main_mask & (self.set_group_ids == int(group)), as_tuple=False
            ).flatten()
            acquisition = self.acquisition_indices[rows]
            if acquisition.unique().numel() != acquisition.numel():
                raise ValueError("acquisition_indices must be unique within each GTN group.")
            rows = rows[torch.argsort(acquisition, stable=True)]
            counts = [
                int((self.stimulus_digits[rows] == digit).sum()) for digit in self.digit_vocab
            ]
            all_counts[int(group)] = counts
            available_ks = [k for k in self.evidence_ks if min(counts) >= k]
            if available_ks:
                largest_k = available_ks[-1]
                self._largest_available_k[int(group)] = largest_k
                checkpoints = [
                    torch.nonzero(self.stimulus_digits[rows] == digit, as_tuple=False).flatten()[
                        largest_k - 1
                    ]
                    for digit in self.digit_vocab
                ]
                checkpoint = int(torch.stack(checkpoints).max())
                sequence = rows[: checkpoint + 1]
                self._sequences[int(group)] = sequence
                unit_digits = self.stimulus_digits[sequence]
                repetition = torch.full_like(sequence, -1)
                for digit in self.digit_vocab:
                    positions = torch.nonzero(
                        unit_digits == digit, as_tuple=False
                    ).flatten()
                    repetition[positions] = torch.arange(
                        positions.numel(), dtype=torch.long, device=sequence.device
                    )
                self._repetition_ranks[int(group)] = repetition
                self._sequence_ranks[int(group)] = torch.arange(
                    sequence.numel(), dtype=torch.long, device=sequence.device
                )
        if not self._sequences:
            raise ValueError(
                "No group has a complete GTN acquisition prefix for the smallest "
                f"requested K={self.evidence_ks[0]}."
            )
        self.n_groups_total = len(group_values)
        self.n_groups_eligible = len(self._sequences)
        self.n_sets_per_epoch = len(self._sequences)
        self.coverage_by_k = {
            k: sum(min(counts) >= k for counts in all_counts.values()) / max(self.n_groups_total, 1)
            for k in self.evidence_ks
        }

        aux_mask = ~main_mask
        self._aux_indices = torch.nonzero(aux_mask, as_tuple=False).flatten()
        set_width = max(int(unit.numel()) for unit in self._sequences.values())
        main_budget = self.batch_size // 2 if self._aux_indices.numel() else self.batch_size
        self.groups_per_batch = max(1, main_budget // set_width)

    @property
    def set_coverage(self) -> float:
        return self.coverage_by_k[self.evidence_k]

    def __len__(self) -> int:
        n = self.n_sets_per_epoch
        return (n + self.groups_per_batch - 1) // self.groups_per_batch

    @property
    def full_context(self) -> TrialContext:
        return TrialContext(
            X=self.X,
            y=self.y,
            domain_id=self.domain_id,
            channel_mask=self.channel_mask,
            prevalidated=self.finite_validated,
        )

    def __iter__(self) -> Iterator[TrialContext]:
        epoch_offset = self._epoch if self.shuffle else 0
        random_device = self.X.device if self.X.device.type == "cuda" else torch.device("cpu")
        gen = torch.Generator(device=random_device).manual_seed(self.seed + epoch_offset)
        if self.shuffle:
            self._epoch += 1
        set_units = list(self._sequences.items())
        if self.shuffle:
            order = torch.randperm(len(set_units), generator=gen, device=random_device).tolist()
            set_units = [set_units[i] for i in order]

        for start in range(0, len(set_units), self.groups_per_batch):
            selected = set_units[start : start + self.groups_per_batch]
            selected_units = [unit for _, unit in selected]
            batch_kmax = min(self._largest_available_k[group] for group, _ in selected)
            indices = torch.cat(selected_units)
            batch_groups = torch.cat(
                [
                    torch.full((unit.numel(),), local_group, dtype=torch.long, device=unit.device)
                    for local_group, (_, unit) in enumerate(selected)
                ]
            )
            batch_ranks = torch.cat([self._repetition_ranks[group] for group, _ in selected])
            batch_sequence_ranks = torch.cat(
                [self._sequence_ranks[group] for group, _ in selected]
            )

            if self._aux_indices.numel() > 0 and indices.numel() < self.batch_size:
                n_aux = min(self.batch_size - indices.numel(), self._aux_indices.numel())
                aux_choice = torch.randperm(
                    self._aux_indices.numel(), generator=gen, device=random_device
                )[:n_aux]
                aux_indices = self._aux_indices[aux_choice.to(self._aux_indices.device)]
                indices = torch.cat([indices, aux_indices])
                batch_groups = torch.cat(
                    [
                        batch_groups,
                        torch.full(
                            (aux_indices.numel(),),
                            -1,
                            dtype=torch.long,
                            device=batch_groups.device,
                        ),
                    ]
                )
                batch_ranks = torch.cat(
                    [
                        batch_ranks,
                        torch.full(
                            (aux_indices.numel(),),
                            -1,
                            dtype=torch.long,
                            device=batch_ranks.device,
                        ),
                    ]
                )
                batch_sequence_ranks = torch.cat(
                    [
                        batch_sequence_ranks,
                        torch.full(
                            (aux_indices.numel(),),
                            -1,
                            dtype=torch.long,
                            device=batch_sequence_ranks.device,
                        ),
                    ]
                )
            if self.shuffle:
                batch_order = torch.randperm(
                    indices.numel(), generator=gen, device=random_device
                )
                if batch_order.device != indices.device:
                    batch_order = batch_order.to(indices.device)
                indices = indices[batch_order]
                batch_groups = batch_groups[batch_order]
                batch_ranks = batch_ranks[batch_order]
                batch_sequence_ranks = batch_sequence_ranks[batch_order]

            yield TrialContext(
                X=self.X[indices],
                y=self.y[indices],
                domain_id=self.domain_id[indices] if self.domain_id is not None else None,
                set_metadata=SetMetadata(
                    stimulus_digits=self.stimulus_digits[indices],
                    group_ids=batch_groups,
                    repetition_ranks=batch_ranks,
                    sequence_ranks=batch_sequence_ranks,
                    prevalidated=True,
                    prevalidated_kmax=batch_kmax,
                ),
                channel_mask=(
                    self.channel_mask[indices]
                    if self.channel_mask is not None and self.channel_mask.dim() == 2
                    else self.channel_mask
                ),
                prevalidated=self.finite_validated,
            )
