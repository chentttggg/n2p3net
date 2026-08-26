"""Alternating optimization for heterogeneous montage branches.

Each batch remains montage-homogeneous, while every branch uses the same
optimizer and the tied canonical backbone in :class:`MultiMontageN2P3Net`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch

from models.multidataset import MultiMontageN2P3Net
from train.contracts import TrialContext
from train.device import empty_cache, get_device
from train.trainer import Trainer, TrainerConfig


@dataclass(frozen=True)
class MultiDatasetSchedule:
    """Batch scheduling policy for one joint epoch.

    ``balanced`` restarts shorter loaders until every domain contributes as
    many batches as the largest loader. ``proportional`` visits every original
    batch exactly once. A positive ``steps_per_epoch`` truncates either policy.
    """

    sampling: str = "balanced"
    steps_per_epoch: int | None = None

    def validate(self) -> None:
        if self.sampling not in {"balanced", "proportional"}:
            raise ValueError("sampling must be 'balanced' or 'proportional'.")
        if self.steps_per_epoch is not None and self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive when provided.")


class MultiDatasetTrainer:
    """Train arbitrary montages with one shared AdamW optimizer.

    The existing :class:`Trainer` remains the source of truth for trial losses,
    augmentation, reconstruction, and variance schedules. This class only
    stamps trusted dataset ids, alternates branch-specific loaders, coordinates
    gradient accumulation, and performs joint early stopping.
    """

    def __init__(
        self,
        model: MultiMontageN2P3Net,
        config: TrainerConfig,
        *,
        channel_masks: Mapping[str, torch.Tensor] | None = None,
        channel_embeddings: Mapping[str, torch.Tensor] | None = None,
        subject_embeddings: Mapping[str, torch.Tensor] | None = None,
        active_domains: Sequence[str] | None = None,
        schedule: MultiDatasetSchedule | None = None,
        device: torch.device | None = None,
    ) -> None:
        if schedule is None:
            schedule = MultiDatasetSchedule()
        schedule.validate()
        if config.lambda4 > 0.0:
            raise ValueError(
                "Marginal MMD cannot be estimated from montage-homogeneous alternating "
                "batches. Keep lambda4=0 (the registered default ablation) or use an "
                "explicit paired-domain feature objective."
            )
        self.model = model
        self.cfg = config
        self.schedule = schedule
        self.active_domains = tuple(active_domains or self.model.domain_names)
        if not self.active_domains or len(set(self.active_domains)) != len(self.active_domains):
            raise ValueError("active_domains must be non-empty and unique.")
        unknown_active = set(self.active_domains) - set(self.model.domain_names)
        if unknown_active:
            raise ValueError(f"active_domains contains unknown domains: {sorted(unknown_active)}.")
        self.device = device if device is not None else get_device()
        self.model.to(self.device)
        self._validate_mapping("channel_masks", channel_masks)
        self._validate_mapping("channel_embeddings", channel_embeddings)
        self._validate_mapping("subject_embeddings", subject_embeddings)

        masks = dict(channel_masks or {})
        channel_features = dict(channel_embeddings or {})
        subject_features = dict(subject_embeddings or {})
        self.branch_trainers = {
            domain: Trainer(
                self.model.branch(domain),
                config,
                E_chn=channel_features.get(domain),
                E_sub=subject_features.get(domain),
                channel_mask=masks.get(domain),
                device=self.device,
            )
            for domain in self.model.domain_names
        }
        active_domain_indices = tuple(
            self.model.domain_index[domain] for domain in self.active_domains
        )
        for trainer in self.branch_trainers.values():
            trainer.active_domain_indices = active_domain_indices

        # Trainer instances must not maintain separate Adam moments for tied
        # parameters. Replacing their empty optimizers with this unique-parameter
        # optimizer gives every domain update the same state trajectory.
        decoder_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "component_decoder" in name
        ]
        decay_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
            and "tau0" not in name
            and "component_decoder" not in name
        ]
        no_decay_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "tau0" in name
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_parameters, "weight_decay": config.weight_decay},
                {"params": no_decay_parameters, "weight_decay": 0.0},
                {
                    "params": decoder_parameters,
                    "weight_decay": config.weight_decay,
                    "lr": config.lr * config.erp_decoder_lr_multiplier,
                },
            ],
            lr=config.lr,
            fused=self.device.type == "cuda",
        )
        for trainer in self.branch_trainers.values():
            trainer.optimizer = self.optimizer

        inactive_indices = sorted(
            set(range(len(self.model.domain_names)))
            - {self.model.domain_index[domain] for domain in self.active_domains}
        )
        self._inactive_domain_rows: list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]] = []
        if inactive_indices:
            row_index = torch.as_tensor(inactive_indices, device=self.device, dtype=torch.long)
            domain_row_names = (
                "dataset_adapter.down",
                "dataset_adapter.up",
                "dom_scale",
                "dom_shift",
                "reference.w_logits",
                "reference.gate_raw",
                "shared_private_encoder.domain_classifier.weight",
                "shared_private_encoder.domain_classifier.bias",
                "shared_private_encoder.dataset_classifier.weight",
                "shared_private_encoder.dataset_classifier.bias",
            )
            for name, parameter in self.model.named_parameters():
                if (
                    parameter.dim() >= 1
                    and parameter.shape[0] == len(self.model.domain_names)
                    and any(token in name for token in domain_row_names)
                ):
                    self._inactive_domain_rows.append(
                        (parameter, row_index, parameter.detach()[row_index].clone())
                    )

        if not 0 <= config.main_domain < len(self.model.domain_names):
            raise ValueError("main_domain is outside the configured montage branches.")
        self.main_domain = self.model.domain_names[config.main_domain]
        if self.main_domain not in self.active_domains:
            raise ValueError("main_domain must be included in active_domains.")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_val_loss = float("inf")
        self.best_epoch: int | None = None

    def _validate_mapping(self, name: str, value: Mapping[str, object] | None) -> None:
        unknown = set(value or {}) - set(self.model.domain_names)
        if unknown:
            raise ValueError(f"{name} contains unknown domains: {sorted(unknown)}.")

    def _with_domain(self, domain: str, batch) -> TrialContext:
        context = Trainer._unpack_batch(batch)
        domain_index = self.model.domain_index[domain]
        if context.domain_id is not None:
            supplied = context.domain_id.reshape(-1)
            if bool((supplied != domain_index).any()):
                raise ValueError(
                    f"Loader {domain!r} supplied domain ids inconsistent with its branch."
                )
        stamped = TrialContext(
            X=context.X,
            y=context.y,
            domain_id=torch.full(
                (context.X.shape[0],),
                domain_index,
                device=context.X.device,
                dtype=torch.long,
            ),
            set_metadata=context.set_metadata,
            channel_mask=context.channel_mask,
        )
        stamped.validate()
        return stamped

    @staticmethod
    def _without_domain(context: TrialContext) -> TrialContext:
        output = TrialContext(
            X=context.X,
            y=context.y,
            set_metadata=context.set_metadata,
            channel_mask=context.channel_mask,
        )
        output.validate()
        return output

    def _validate_loaders(self, name: str, loaders: Mapping[str, Iterable]) -> None:
        expected = set(self.active_domains)
        actual = set(loaders)
        if actual != expected:
            raise ValueError(
                f"{name} must contain exactly {sorted(expected)}, got {sorted(actual)}."
            )
        if any(len(loader) < 1 for loader in loaders.values()):
            raise ValueError(f"Every {name} loader must contain at least one batch.")

    def _trial_batches_per_epoch(self, loaders: Mapping[str, Iterable]) -> int:
        lengths = [len(loaders[domain]) for domain in self.active_domains]
        count = (
            max(lengths) * len(lengths)
            if self.schedule.sampling == "balanced"
            else sum(lengths)
        )
        if self.schedule.steps_per_epoch is not None:
            count = min(count, self.schedule.steps_per_epoch)
        return count

    def _alternating_batches(
        self,
        loaders: Mapping[str, Iterable],
        *,
        epoch: int,
    ) -> Iterator[tuple[str, object]]:
        domains = list(self.active_domains)
        rotation = epoch % len(domains)
        domains = domains[rotation:] + domains[:rotation]
        lengths = {domain: len(loaders[domain]) for domain in domains}
        if self.schedule.sampling == "balanced":
            rounds = max(lengths.values())
            iterators = {domain: iter(loaders[domain]) for domain in domains}
            emitted = 0
            for _ in range(rounds):
                for domain in domains:
                    try:
                        batch = next(iterators[domain])
                    except StopIteration:
                        iterators[domain] = iter(loaders[domain])
                        batch = next(iterators[domain])
                    yield domain, batch
                    emitted += 1
                    if (
                        self.schedule.steps_per_epoch is not None
                        and emitted >= self.schedule.steps_per_epoch
                    ):
                        return
            return

        iterators = {domain: iter(loaders[domain]) for domain in domains}
        active = set(domains)
        emitted = 0
        while active:
            for domain in domains:
                if domain not in active:
                    continue
                try:
                    batch = next(iterators[domain])
                except StopIteration:
                    active.remove(domain)
                    continue
                yield domain, batch
                emitted += 1
                if (
                    self.schedule.steps_per_epoch is not None
                    and emitted >= self.schedule.steps_per_epoch
                ):
                    return

    def _clamp_shared_tau(self) -> None:
        for domain in self.model.domain_names:
            self.model.branch(domain).component_window.clamp_tau0_()

    def _set_accumulation_state(self, value: bool) -> None:
        for trainer in self.branch_trainers.values():
            trainer._accum_has_grad = value

    @torch.no_grad()
    def _restore_inactive_domain_rows(self) -> None:
        """Keep held-out adapter/FiLM rows at their exact initialization."""

        for parameter, rows, initial in self._inactive_domain_rows:
            parameter[rows] = initial

    def _domain_loader(self, domain: str, loader: Iterable) -> Iterator[TrialContext]:
        for batch in loader:
            yield self._with_domain(domain, batch)

    def _prepare_profiles(
        self,
        reconstruction_contexts: Mapping[str, TrialContext] | None,
        train_set_loaders: Mapping[str, Iterable] | None,
    ) -> None:
        contexts = dict(reconstruction_contexts or {})
        if self.cfg.lambda_recon > 0.0 and set(contexts) != set(self.active_domains):
            raise ValueError(
                "lambda_recon>0 requires fold-training reconstruction_contexts for every domain."
            )
        for domain in self.active_domains:
            trainer = self.branch_trainers[domain]
            context = contexts.get(domain)
            trainer._prepare_reconstruction_profile(
                self._without_domain(context) if context is not None else None
            )

        if self.cfg.lambda_conditional_nll > 0.0:
            set_loader = (train_set_loaders or {}).get(self.main_domain)
            context = (
                set_loader.full_context
                if set_loader is not None and hasattr(set_loader, "full_context")
                else contexts.get(self.main_domain)
            )
            if context is not None:
                context = self._without_domain(context)
            self.branch_trainers[self.main_domain]._prepare_repetition_calibration(context)

    def _validate_set_loaders(
        self,
        train_set_loaders: Mapping[str, Iterable] | None,
        val_set_loaders: Mapping[str, Iterable] | None,
        *,
        has_validation: bool,
    ) -> None:
        active = self.cfg.lambda_digit > 0.0 or self.cfg.lambda_conditional_nll > 0.0
        train_domains = set(train_set_loaders or {})
        val_domains = set(val_set_loaders or {})
        if active and train_domains != {self.main_domain}:
            raise ValueError("Set objectives require exactly one main-domain train_set_loader.")
        if not active and train_domains:
            raise ValueError("Set loaders were supplied while set objectives are disabled.")
        if has_validation and active and val_domains != {self.main_domain}:
            raise ValueError(
                "Set-supervised validation requires exactly one main-domain val_set_loader."
            )
        if val_domains - {self.main_domain}:
            raise ValueError("Only the main domain may provide repetition set supervision.")

    def fit(
        self,
        train_loaders: Mapping[str, Iterable],
        val_loaders: Mapping[str, Iterable] | None = None,
        *,
        train_set_loaders: Mapping[str, Iterable] | None = None,
        val_set_loaders: Mapping[str, Iterable] | None = None,
        reconstruction_contexts: Mapping[str, TrialContext] | None = None,
    ) -> dict[str, object]:
        """Fit all montage branches using balanced or proportional alternation."""

        self._validate_loaders("train_loaders", train_loaders)
        if val_loaders is not None:
            self._validate_loaders("val_loaders", val_loaders)
        self._validate_set_loaders(
            train_set_loaders,
            val_set_loaders,
            has_validation=val_loaders is not None,
        )
        self._prepare_profiles(reconstruction_contexts, train_set_loaders)

        first_joint_epoch = self.branch_trainers[self.main_domain]._first_joint_epoch()
        if val_loaders is not None and self.cfg.epochs <= first_joint_epoch:
            raise ValueError(
                "Phase-aware early stopping requires at least one joint epoch after ramps."
            )

        train_losses: list[float] = []
        train_losses_by_domain: dict[str, list[float]] = {
            domain: [] for domain in self.active_domains
        }
        val_losses: list[float] = []
        val_losses_by_domain: dict[str, list[float]] = {
            domain: [] for domain in self.active_domains
        }
        val_task_losses_by_domain: dict[str, list[float]] = {
            domain: [] for domain in self.active_domains
        }
        phases: list[str] = []
        patience_left = self.cfg.early_stop_patience
        shared_accumulation = False
        main_trainer = self.branch_trainers[self.main_domain]
        trial_batches = self._trial_batches_per_epoch(train_loaders)
        set_batches = sum(len(loader) for loader in (train_set_loaders or {}).values())
        planned_steps_per_epoch = math.ceil(trial_batches / self.cfg.accum_steps) + set_batches
        main_trainer._configure_lr_scheduler(self.cfg.epochs * planned_steps_per_epoch)
        for trainer in self.branch_trainers.values():
            trainer.lr_scheduler = main_trainer.lr_scheduler
            trainer.planned_optimizer_steps = main_trainer.planned_optimizer_steps

        for epoch in range(self.cfg.epochs):
            epoch_limit = main_trainer._required_epoch_count()
            for trainer in self.branch_trainers.values():
                trainer._active_recon_nll_weight = trainer._variance_nll_weight_for_epoch(epoch)
                trainer._variance_nll_weight_history.append(trainer._active_recon_nll_weight)
            phase = main_trainer._phase_for_epoch(epoch)
            phases.append(phase)
            self.model.train()

            domain_sums = {
                domain: torch.zeros((), device=self.device, dtype=torch.float32)
                for domain in self.active_domains
            }
            domain_counts = dict.fromkeys(self.active_domains, 0)
            try:
                for epoch_step, (domain, batch) in enumerate(
                    self._alternating_batches(train_loaders, epoch=epoch)
                ):
                    trainer = self.branch_trainers[domain]
                    trainer._accum_has_grad = shared_accumulation
                    loss = trainer._train_step(self._with_domain(domain, batch), epoch_step)
                    self._restore_inactive_domain_rows()
                    shared_accumulation = trainer._accum_has_grad
                    self._set_accumulation_state(shared_accumulation)
                    domain_sums[domain] += loss.float()
                    domain_counts[domain] += 1

                for domain, loader in (train_set_loaders or {}).items():
                    trainer = self.branch_trainers[domain]
                    trainer._accum_has_grad = shared_accumulation
                    for batch in loader:
                        weighted, _, _ = trainer._set_train_step(self._with_domain(domain, batch))
                        self._restore_inactive_domain_rows()
                        domain_sums[domain] += weighted.float()
                        domain_counts[domain] += 1
                    shared_accumulation = False
                    self._set_accumulation_state(False)
                    self.optimizer.zero_grad(set_to_none=True)
            except torch.OutOfMemoryError:
                raise RuntimeError(
                    "Out of memory during multi-dataset training; reduce per-domain batch "
                    "sizes or increase accum_steps."
                ) from None

            if shared_accumulation:
                main_trainer._optimizer_step()
                self._restore_inactive_domain_rows()
                self._clamp_shared_tau()
                self.optimizer.zero_grad(set_to_none=True)
                shared_accumulation = False
                self._set_accumulation_state(False)

            domain_means = {
                domain: float((domain_sums[domain] / max(domain_counts[domain], 1)).item())
                for domain in self.active_domains
            }
            for domain, value in domain_means.items():
                train_losses_by_domain[domain].append(value)
            train_loss = sum(domain_means.values()) / len(domain_means)
            train_losses.append(train_loss)

            if val_loaders is None:
                print(
                    f"[epoch {epoch + 1}/{epoch_limit}] train_loss={train_loss:.4f} phase={phase}",
                    flush=True,
                )
                continue

            current_objective_validation = {}
            current_task_validation = {}
            for domain, loader in val_loaders.items():
                trainer = self.branch_trainers[domain]
                objective_value = trainer._evaluate(self._domain_loader(domain, loader))
                task_value = trainer._evaluate_task(self._domain_loader(domain, loader))
                set_loader = (val_set_loaders or {}).get(domain)
                if set_loader is not None:
                    objective_value += trainer._evaluate_set(
                        self._domain_loader(domain, set_loader)
                    )
                    task_value += self.cfg.lambda_digit * trainer._evaluate_set_task(
                        self._domain_loader(domain, set_loader)
                    )
                current_objective_validation[domain] = objective_value
                current_task_validation[domain] = task_value
                val_losses_by_domain[domain].append(objective_value)
                val_task_losses_by_domain[domain].append(task_value)
            val_loss = current_task_validation[self.main_domain]
            val_losses.append(val_loss)
            if phase == "joint":
                if val_loss < self.best_val_loss - 1e-6:
                    self.best_val_loss = val_loss
                    self.best_state = {
                        key: value.detach().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    self.best_epoch = epoch
                    patience_left = self.cfg.early_stop_patience
                else:
                    patience_left -= 1
            print(
                f"[epoch {epoch + 1}/{epoch_limit}] train_loss={train_loss:.4f} "
                f"main_task_val_loss={val_loss:.4f} "
                f"main_objective_val_loss="
                f"{current_objective_validation[self.main_domain]:.4f} phase={phase}",
                flush=True,
            )
            if phase == "joint" and patience_left <= 0:
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        empty_cache(self.device)
        return {
            "train_losses": train_losses,
            "train_losses_by_domain": train_losses_by_domain,
            "val_losses": val_losses,
            "val_losses_by_domain": val_losses_by_domain,
            "val_task_losses_by_domain": val_task_losses_by_domain,
            "selection_domain": self.main_domain,
            "selection_metric": "target_bce_plus_weighted_digit_ce",
            "best_epoch": self.best_epoch,
            "phases": phases,
            "configured_epochs": self.cfg.epochs,
            "effective_epoch_limit": main_trainer._required_epoch_count(),
            "sampling": self.schedule.sampling,
            "active_domains": list(self.active_domains),
        }

    def refit_repetition_evidence(
        self,
        loader: Iterable,
        *,
        epochs: int | None = None,
    ) -> list[float]:
        """Refit the shared repetition density using main-domain complete sets."""

        trainer = self.branch_trainers[self.main_domain]
        return trainer.refit_repetition_evidence(
            self._domain_loader(self.main_domain, loader),
            epochs=epochs,
        )
