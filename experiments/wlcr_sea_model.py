from __future__ import annotations

"""Core WLCR-SEA components for exploratory request-local forecasting.

The module consumes only one request's 336-hour traffic tensor and observation
mask.  It never opens data files, never accepts a cell identifier as a model
feature, and exposes finite seasonal experts plus an inspectable allocation.
"""

from dataclasses import dataclass
import math
import warnings
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from experiments.missingness_protocol import (
    corruption_statistics,
    global_corruption_mask,
    stable_uniform,
)

INPUT_HOURS = 336
FORECAST_HOURS = 24
TARGET_COUNT = 4
EXPERT_COUNT = 8
EXPERT_NAMES = (
    "last_day",
    "last_week",
    "last_biweek",
    "same_hour_median_7d",
    "same_hour_median_14d",
    "bounded_week_trend",
    "window_local_median",
    "training_prior",
)
METRIC_NAMES = (
    "ul_active_users",
    "dl_active_users",
    "dl_prb",
    "ul_prb",
)
STRICT_THRESHOLDS = np.asarray((0.2, 0.3, 0.4, 0.5), dtype=np.float64)
TREND_CLIP = 0.5
EXPERT_DISTANCE_HOURS = (24.0, 168.0, 336.0, 96.0, 180.0, 252.0, 168.0, 0.0)
FIXED_SEASONAL_WEIGHTS = (0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ExpertBatch:
    values: np.ndarray
    availability: np.ndarray
    reliability: np.ndarray
    context: np.ndarray


@dataclass(frozen=True)
class VariantConfig:
    name: str
    attention: str
    hard_mask: bool
    reliability: bool
    residual_bound: float
    augmentation: str = "clean"
    consistency_weight: float = 0.0
    cross_indicator_context: bool = False


VARIANTS: Mapping[str, VariantConfig] = {
    "A0_fixed": VariantConfig("A0_fixed", "fixed", True, False, 0.0),
    "A0_global_static": VariantConfig(
        "A0_global_static", "global_softmax", True, False, 0.0
    ),
    "A0_horizon_indicator": VariantConfig(
        "A0_horizon_indicator", "horizon_softmax", True, False, 0.0
    ),
    "A1_softmax": VariantConfig("A1_softmax", "softmax", False, False, 0.0),
    "A2_entmax": VariantConfig("A2_entmax", "entmax15", False, False, 0.0),
    "A3_hard_mask": VariantConfig("A3_hard_mask", "entmax15", True, False, 0.0),
    "A4_reliability": VariantConfig("A4_reliability", "entmax15", True, True, 0.0),
    "A5_residual": VariantConfig("A5_residual", "entmax15", True, True, 0.5),
    "A6_mcar_aug": VariantConfig(
        "A6_mcar_aug", "entmax15", True, True, 0.5, augmentation="mcar"
    ),
    "A6_block_aug": VariantConfig(
        "A6_block_aug", "entmax15", True, True, 0.5, augmentation="block"
    ),
    "A6_mixed_aug": VariantConfig(
        "A6_mixed_aug", "entmax15", True, True, 0.5, augmentation="mixed"
    ),
    "A6_consistency": VariantConfig(
        "A6_consistency",
        "entmax15",
        True,
        True,
        0.5,
        augmentation="mixed",
        consistency_weight=0.05,
    ),
    "A7_cross_indicator": VariantConfig(
        "A7_cross_indicator",
        "entmax15",
        True,
        True,
        0.5,
        augmentation="mixed",
        consistency_weight=0.05,
        cross_indicator_context=True,
    ),
}


def seasonal_history_index(horizon: int, distance_days: int) -> int:
    """Return the zero-based history index for target horizon/day distance."""
    if not 1 <= horizon <= FORECAST_HOURS:
        raise ValueError("horizon must be in 1..24")
    if not 1 <= distance_days <= 14:
        raise ValueError("distance_days must be in 1..14")
    index = INPUT_HOURS - 1 + horizon - 24 * distance_days
    if not 0 <= index < INPUT_HOURS:
        raise ValueError("seasonal index is outside the request history")
    return index


def _masked_median(
    values: np.ndarray, masks: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(masks, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(masked, axis=axis)
    count = np.sum(masks, axis=axis)
    return median, count


def _safe_context(values: np.ndarray, masks: np.ndarray, prior_log: np.ndarray) -> np.ndarray:
    count = np.sum(masks, axis=1).astype(np.float32)
    safe_count = np.maximum(count, 1.0)
    mean = np.sum(np.where(masks, values, 0.0), axis=1) / safe_count
    median, _ = _masked_median(values, masks, axis=1)
    centered = np.where(masks, values - mean[:, None, :], 0.0)
    std = np.sqrt(np.sum(centered * centered, axis=1) / safe_count)
    fallback = np.mean(prior_log, axis=0)[None, :]
    mean = np.where(count > 0, mean, fallback)
    median = np.where(count > 0, median, fallback)
    std = np.where(count > 1, std, 0.0)
    recent, recent_count = _masked_median(values[:, -24:, :], masks[:, -24:, :], axis=1)
    previous, previous_count = _masked_median(
        values[:, -48:-24, :], masks[:, -48:-24, :], axis=1
    )
    trend = np.where(
        (recent_count > 0) & (previous_count > 0), recent - previous, 0.0
    )
    missing_fraction = 1.0 - count / float(INPUT_HOURS)
    return np.stack((mean, median, std, missing_fraction, trend), axis=-1).astype(
        np.float32
    )


def build_expert_batch(
    x_values: np.ndarray,
    x_masks: np.ndarray,
    prior_log: np.ndarray,
    *,
    additional_missing: np.ndarray | None = None,
    trend_clip: float = TREND_CLIP,
) -> ExpertBatch:
    """Build eight experts using only each supplied request window.

    ``x_values`` are log1p values with arbitrary finite fill values; masks are
    authoritative. Artificial removals are applied to masks before every
    aggregate so deleted observations cannot leak through filled values.
    """
    values = np.asarray(x_values, dtype=np.float32)
    original_masks = np.asarray(x_masks, dtype=bool)
    prior = np.asarray(prior_log, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (INPUT_HOURS, TARGET_COUNT):
        raise ValueError(f"unexpected history values shape: {values.shape}")
    if original_masks.shape != values.shape:
        raise ValueError("history masks must match values")
    if prior.shape != (FORECAST_HOURS, TARGET_COUNT):
        raise ValueError(f"unexpected prior shape: {prior.shape}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(prior)):
        raise ValueError("expert inputs and prior must be finite")
    if additional_missing is None:
        masks = original_masks.copy()
    else:
        removed = np.asarray(additional_missing, dtype=bool)
        if removed.shape != values.shape:
            raise ValueError("additional missing mask must match histories")
        masks = original_masks & ~removed

    n = values.shape[0]
    expert_values = np.empty(
        (n, FORECAST_HOURS, TARGET_COUNT, EXPERT_COUNT), dtype=np.float32
    )
    availability = np.zeros_like(expert_values, dtype=bool)
    reliability = np.zeros_like(expert_values, dtype=np.float32)
    prior_full = np.broadcast_to(prior[None, :, :], (n, FORECAST_HOURS, TARGET_COUNT))

    single_days = (1, 7, 14)
    for expert, days in enumerate(single_days):
        indices = np.asarray(
            [seasonal_history_index(horizon, days) for horizon in range(1, 25)]
        )
        selected_values = values[:, indices, :]
        selected_masks = masks[:, indices, :]
        expert_values[..., expert] = np.where(
            selected_masks, selected_values, prior_full
        )
        availability[..., expert] = selected_masks
        reliability[..., expert] = selected_masks.astype(np.float32)

    for expert, days in ((3, 7), (4, 14)):
        indices = np.asarray(
            [
                [seasonal_history_index(horizon, day) for day in range(1, days + 1)]
                for horizon in range(1, 25)
            ]
        )
        selected_values = values[:, indices, :]
        selected_masks = masks[:, indices, :]
        medians, counts = _masked_median(selected_values, selected_masks, axis=2)
        valid = counts > 0
        expert_values[..., expert] = np.where(valid, medians, prior_full)
        availability[..., expert] = valid
        reliability[..., expert] = counts.astype(np.float32) / float(days)

    week = expert_values[..., 1]
    biweek = expert_values[..., 2]
    trend_available = availability[..., 1] & availability[..., 2]
    trend = week + np.clip(week - biweek, -trend_clip, trend_clip)
    expert_values[..., 5] = np.where(trend_available, trend, prior_full)
    availability[..., 5] = trend_available
    reliability[..., 5] = np.where(
        trend_available,
        np.minimum(reliability[..., 1], reliability[..., 2]),
        0.0,
    )

    local_median, local_count = _masked_median(values, masks, axis=1)
    local_valid = local_count > 0
    local_full = np.broadcast_to(
        local_median[:, None, :], (n, FORECAST_HOURS, TARGET_COUNT)
    )
    local_valid_full = np.broadcast_to(
        local_valid[:, None, :], (n, FORECAST_HOURS, TARGET_COUNT)
    )
    local_reliability = np.broadcast_to(
        (local_count.astype(np.float32) / float(INPUT_HOURS))[:, None, :],
        (n, FORECAST_HOURS, TARGET_COUNT),
    )
    expert_values[..., 6] = np.where(local_valid_full, local_full, prior_full)
    availability[..., 6] = local_valid_full
    reliability[..., 6] = np.where(local_valid_full, local_reliability, 0.0)

    expert_values[..., 7] = prior_full
    availability[..., 7] = True
    reliability[..., 7] = 1.0
    context = _safe_context(values, masks, prior)
    if not np.all(np.isfinite(expert_values)) or not np.all(np.isfinite(context)):
        raise ValueError("expert construction produced non-finite values")
    return ExpertBatch(expert_values, availability, reliability, context)


def training_prior_log(
    targets: np.ndarray, target_masks: np.ndarray, indices: Sequence[int]
) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float64)[np.asarray(indices, dtype=np.int64)]
    masks = np.asarray(target_masks, dtype=bool)[np.asarray(indices, dtype=np.int64)]
    if values.shape[1:] != (FORECAST_HOURS, TARGET_COUNT):
        raise ValueError("unexpected target shape for training prior")
    transformed = np.where(masks, np.log1p(np.maximum(values, 0.0)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        prior = np.nanmedian(transformed, axis=0)
    if not np.all(np.isfinite(prior)):
        raise ValueError("training layer cannot estimate every prior")
    return prior.astype(np.float32)


class _Entmax15Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, dim: int) -> torch.Tensor:
        moved = logits.movedim(dim, -1)
        scaled = moved / 2.0
        scaled = scaled - scaled.max(dim=-1, keepdim=True).values
        sorted_values, _ = torch.sort(scaled, dim=-1, descending=True)
        cumulative = torch.stack(
            [sorted_values[..., :width].sum(dim=-1) for width in range(1, sorted_values.shape[-1] + 1)],
            dim=-1,
        )
        cumulative_sq = torch.stack(
            [sorted_values[..., :width].square().sum(dim=-1) for width in range(1, sorted_values.shape[-1] + 1)],
            dim=-1,
        )
        rho = torch.arange(
            1, sorted_values.shape[-1] + 1, device=scaled.device, dtype=scaled.dtype
        )
        mean = cumulative / rho
        mean_sq = cumulative_sq / rho
        ss = rho * (mean_sq - mean.square())
        delta = torch.clamp((1.0 - ss) / rho, min=0.0)
        tau = mean - torch.sqrt(delta)
        support = (tau <= sorted_values).sum(dim=-1, keepdim=True).clamp(min=1)
        tau_star = tau.gather(-1, support - 1)
        probabilities = torch.clamp(scaled - tau_star, min=0.0).square()
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probabilities = probabilities.movedim(-1, dim)
        ctx.dim = dim
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (probabilities,) = ctx.saved_tensors
        gppr = torch.sqrt(torch.clamp(probabilities, min=0.0))
        grad = grad_output * gppr
        correction = grad.sum(dim=ctx.dim, keepdim=True) / gppr.sum(
            dim=ctx.dim, keepdim=True
        ).clamp_min(1e-12)
        grad = grad - correction * gppr
        return grad, None


def entmax15(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable alpha=1.5 Entmax with its analytic backward."""
    return _Entmax15Function.apply(logits, dim)


def available_set_entmax15(
    logits: torch.Tensor, availability: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Apply Entmax exactly on each row's available expert subset.

    This deliberately does not emulate masking by assigning a finite low score
    to unavailable experts. Each compacted row contains only the logits in its
    available set, Entmax is evaluated on that set, and the resulting weights
    are scattered back into the original expert axis. Consequently,
    unavailable experts are structurally assigned exact zero mass.
    """
    if logits.shape != availability.shape:
        raise ValueError("logits and availability must have the same shape")
    if dim < 0:
        dim += logits.ndim
    if not 0 <= dim < logits.ndim:
        raise ValueError("attention dimension is outside the tensor rank")
    moved_logits = logits.movedim(dim, -1)
    moved_available = availability.bool().movedim(dim, -1)
    expert_count = moved_logits.shape[-1]
    flat_logits = moved_logits.reshape(-1, expert_count)
    flat_available = moved_available.reshape(-1, expert_count)
    counts = flat_available.sum(dim=-1)
    if bool(torch.any(counts == 0)):
        raise ValueError("hard-masked routing requires at least one available expert")

    result = torch.zeros_like(flat_logits)
    # There are only eight experts. Grouping rows by available-set size keeps
    # the operation vectorized while preserving exact subset semantics.
    for subset_size in range(1, expert_count + 1):
        row_selector = counts == subset_size
        if not bool(torch.any(row_selector)):
            continue
        row_indices = torch.nonzero(row_selector, as_tuple=False).squeeze(-1)
        local_logits = flat_logits.index_select(0, row_indices)
        local_available = flat_available.index_select(0, row_indices)
        compact_logits = local_logits[local_available].reshape(-1, subset_size)
        compact_attention = entmax15(compact_logits, dim=-1)
        local_attention = torch.zeros_like(local_logits).masked_scatter(
            local_available, compact_attention.reshape(-1)
        )
        result = result.index_copy(0, row_indices, local_attention)
    return result.reshape_as(moved_logits).movedim(-1, dim)


class WLCRSEA(nn.Module):
    """Single-head sparse attention over a finite seasonal expert set."""

    def __init__(
        self,
        variant: VariantConfig,
        *,
        token_dim: int = 32,
        hidden_dim: int = 64,
        residual_bound: float | None = None,
    ) -> None:
        super().__init__()
        if token_dim not in (16, 32):
            raise ValueError("token_dim must be 16 or 32")
        self.variant = variant
        self.token_dim = token_dim
        self.residual_bound = (
            variant.residual_bound if residual_bound is None else residual_bound
        )
        if self.residual_bound < 0.0:
            raise ValueError("residual bound must be non-negative")
        self.embedding_dim = max(4, token_dim // 4)
        self.context_dim = 5 * (
            TARGET_COUNT if variant.cross_indicator_context else 1
        )
        self.register_buffer(
            "fixed_weights", torch.tensor(FIXED_SEASONAL_WEIGHTS, dtype=torch.float32)
        )

        if variant.attention == "fixed":
            if self.residual_bound != 0.0:
                raise ValueError("fixed attention does not support a learned residual")
            return
        if variant.attention == "global_softmax":
            if self.residual_bound != 0.0:
                raise ValueError("global static routing does not support a residual")
            self.global_indicator_logits = nn.Parameter(
                torch.zeros(TARGET_COUNT, EXPERT_COUNT)
            )
            return
        if variant.attention == "horizon_softmax":
            if self.residual_bound != 0.0:
                raise ValueError("horizon-indicator routing does not support a residual")
            self.horizon_indicator_logits = nn.Parameter(
                torch.zeros(FORECAST_HOURS, TARGET_COUNT, EXPERT_COUNT)
            )
            return
        if variant.attention not in {"softmax", "entmax15"}:
            raise ValueError(f"unknown attention function: {variant.attention}")

        self.horizon_embedding = nn.Embedding(FORECAST_HOURS, self.embedding_dim)
        self.metric_embedding = nn.Embedding(TARGET_COUNT, self.embedding_dim)
        self.expert_embedding = nn.Embedding(EXPERT_COUNT, self.embedding_dim)
        query_input = 2 * self.embedding_dim + self.context_dim
        scalar_features = 4 if variant.reliability else 3
        key_input = self.embedding_dim + scalar_features
        self.query_network = nn.Sequential(
            nn.Linear(query_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.key_network = nn.Sequential(
            nn.Linear(key_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        if variant.reliability:
            self.reliability_beta_raw = nn.Parameter(torch.tensor(0.0))
        if self.residual_bound > 0.0:
            residual_input = 1 + self.context_dim + 2 * self.embedding_dim + 1
            self.residual_network = nn.Sequential(
                nn.Linear(residual_input, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        distance = torch.tensor(EXPERT_DISTANCE_HOURS, dtype=torch.float32) / float(
            INPUT_HOURS
        )
        self.register_buffer("expert_distance", distance)

    def _context_for_queries(self, context: torch.Tensor) -> torch.Tensor:
        batch = context.shape[0]
        if self.variant.cross_indicator_context:
            shared = context.reshape(batch, -1)
            return shared[:, None, None, :].expand(
                batch, FORECAST_HOURS, TARGET_COUNT, shared.shape[-1]
            )
        return context[:, None, :, :].expand(
            batch, FORECAST_HOURS, TARGET_COUNT, context.shape[-1]
        )

    def _attention(
        self,
        values: torch.Tensor,
        availability: torch.Tensor,
        reliability: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant.attention == "fixed":
            weights = self.fixed_weights.view(1, 1, 1, EXPERT_COUNT)
            weights = weights * availability.to(values.dtype)
            total = weights.sum(dim=-1, keepdim=True)
            fallback = torch.zeros_like(weights)
            fallback[..., 7] = 1.0
            return torch.where(total > 0.0, weights / total.clamp_min(1e-12), fallback)
        if self.variant.attention == "global_softmax":
            logits = self.global_indicator_logits.view(
                1, 1, TARGET_COUNT, EXPERT_COUNT
            ).expand(values.shape[0], FORECAST_HOURS, TARGET_COUNT, EXPERT_COUNT)
            if bool(torch.any(~availability.any(dim=-1))):
                raise ValueError("hard-masked routing requires an available expert")
            # Softmax is evaluated on the available set; -inf is the exact
            # exclusion sentinel supported by torch.softmax.
            return torch.softmax(logits.masked_fill(~availability, -torch.inf), dim=-1)
        if self.variant.attention == "horizon_softmax":
            logits = self.horizon_indicator_logits.unsqueeze(0).expand(
                values.shape[0], FORECAST_HOURS, TARGET_COUNT, EXPERT_COUNT
            )
            if bool(torch.any(~availability.any(dim=-1))):
                raise ValueError("hard-masked routing requires an available expert")
            return torch.softmax(logits.masked_fill(~availability, -torch.inf), dim=-1)
        batch = values.shape[0]
        h_index = torch.arange(FORECAST_HOURS, device=values.device)
        q_index = torch.arange(TARGET_COUNT, device=values.device)
        j_index = torch.arange(EXPERT_COUNT, device=values.device)
        h_embed = self.horizon_embedding(h_index)[None, :, None, :].expand(
            batch, FORECAST_HOURS, TARGET_COUNT, -1
        )
        q_embed = self.metric_embedding(q_index)[None, None, :, :].expand(
            batch, FORECAST_HOURS, TARGET_COUNT, -1
        )
        query_context = self._context_for_queries(context)
        query = self.query_network(torch.cat((h_embed, q_embed, query_context), dim=-1))
        type_embed = self.expert_embedding(j_index)[None, None, None, :, :].expand(
            batch, FORECAST_HOURS, TARGET_COUNT, EXPERT_COUNT, -1
        )
        distance = self.expert_distance.view(1, 1, 1, EXPERT_COUNT, 1).expand(
            batch, FORECAST_HOURS, TARGET_COUNT, EXPERT_COUNT, 1
        )
        scalars = [
            values.unsqueeze(-1),
            availability.to(values.dtype).unsqueeze(-1),
            distance,
        ]
        if self.variant.reliability:
            scalars.append(reliability.unsqueeze(-1))
        key = self.key_network(torch.cat((type_embed, *scalars), dim=-1))
        logits = torch.sum(query.unsqueeze(-2) * key, dim=-1) / math.sqrt(self.token_dim)
        if self.variant.reliability:
            beta = torch.nn.functional.softplus(self.reliability_beta_raw)
            logits = logits + beta * torch.log(reliability.clamp_min(1e-6))
        if self.variant.hard_mask:
            if self.variant.attention == "softmax":
                if bool(torch.any(~availability.any(dim=-1))):
                    raise ValueError("hard-masked routing requires an available expert")
                attention = torch.softmax(
                    logits.masked_fill(~availability, -torch.inf), dim=-1
                )
            elif self.variant.attention == "entmax15":
                attention = available_set_entmax15(logits, availability, dim=-1)
            else:
                raise ValueError(f"unknown attention function: {self.variant.attention}")
        elif self.variant.attention == "softmax":
            attention = torch.softmax(logits, dim=-1)
        elif self.variant.attention == "entmax15":
            attention = entmax15(logits, dim=-1)
        else:
            raise ValueError(f"unknown attention function: {self.variant.attention}")
        return attention

    def forward(
        self,
        values: torch.Tensor,
        availability: torch.Tensor,
        reliability: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if values.shape[1:] != (FORECAST_HOURS, TARGET_COUNT, EXPERT_COUNT):
            raise ValueError(f"unexpected expert tensor shape: {tuple(values.shape)}")
        attention = self._attention(values, availability.bool(), reliability, context)
        baseline = torch.sum(attention * values, dim=-1)
        entropy = -torch.sum(attention * torch.log(attention.clamp_min(1e-12)), dim=-1)
        residual = torch.zeros_like(baseline)
        if self.residual_bound > 0.0:
            batch = values.shape[0]
            h_index = torch.arange(FORECAST_HOURS, device=values.device)
            q_index = torch.arange(TARGET_COUNT, device=values.device)
            h_embed = self.horizon_embedding(h_index)[None, :, None, :].expand(
                batch, FORECAST_HOURS, TARGET_COUNT, -1
            )
            q_embed = self.metric_embedding(q_index)[None, None, :, :].expand(
                batch, FORECAST_HOURS, TARGET_COUNT, -1
            )
            query_context = self._context_for_queries(context)
            residual_input = torch.cat(
                (
                    baseline.unsqueeze(-1),
                    query_context,
                    h_embed,
                    q_embed,
                    entropy.unsqueeze(-1),
                ),
                dim=-1,
            )
            residual = self.residual_bound * torch.tanh(
                self.residual_network(residual_input).squeeze(-1)
            )
        prediction_log = baseline + residual
        return {
            "prediction_log": prediction_log,
            "baseline_log": baseline,
            "residual": residual,
            "attention": attention,
            "entropy": entropy,
        }


def sea_loss(
    output: Mapping[str, torch.Tensor],
    target_log: torch.Tensor,
    target_mask: torch.Tensor,
    reliability: torch.Tensor,
    *,
    residual_weight: float = 1e-3,
    reliability_weight: float = 1e-3,
    consistency_output: Mapping[str, torch.Tensor] | None = None,
    consistency_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = target_mask.to(target_log.dtype)
    denominator = mask.sum().clamp_min(1.0)
    prediction = output["prediction_log"]
    prediction_loss = (torch.abs(prediction - target_log) * mask).sum() / denominator
    residual_loss = (output["residual"].square() * mask).sum() / denominator
    reliability_loss = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    if reliability_weight > 0.0:
        reliability_loss = torch.mean(output["attention"] * (1.0 - reliability))
    consistency_loss = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    if consistency_output is not None and consistency_weight > 0.0:
        consistency_loss = (
            torch.abs(
                consistency_output["prediction_log"] - prediction.detach()
            )
            * mask
        ).sum() / denominator
    total = (
        prediction_loss
        + residual_weight * residual_loss
        + reliability_weight * reliability_loss
        + consistency_weight * consistency_loss
    )
    return total, {
        "prediction": float(prediction_loss.detach().cpu()),
        "residual": float(residual_loss.detach().cpu()),
        "reliability": float(reliability_loss.detach().cpu()),
        "consistency": float(consistency_loss.detach().cpu()),
    }


def prediction_from_log(values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(values, torch.Tensor):
        return torch.clamp(torch.expm1(values), min=1e-4)
    return np.maximum(np.expm1(np.asarray(values)), 1e-4).astype(np.float32)



def bounded_audit_envelope(
    expert_values: np.ndarray,
    availability: np.ndarray,
    residual_bound: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the guaranteed log-forecast envelope for available experts."""
    values = np.asarray(expert_values, dtype=np.float64)
    available = np.asarray(availability, dtype=bool)
    if values.shape != available.shape or values.shape[-1] != EXPERT_COUNT:
        raise ValueError("audit envelope requires aligned expert tensors")
    if residual_bound < 0.0:
        raise ValueError("residual_bound must be non-negative")
    if np.any(~np.any(available, axis=-1)):
        raise ValueError("every forecast must have at least one available expert")
    lower = np.min(np.where(available, values, np.inf), axis=-1) - residual_bound
    upper = np.max(np.where(available, values, -np.inf), axis=-1) + residual_bound
    return lower, upper


def frozen_low_activity_thresholds(
    targets: np.ndarray, target_masks: np.ndarray, train_indices: Sequence[int]
) -> np.ndarray:
    actual = np.asarray(targets, dtype=np.float64)[np.asarray(train_indices, dtype=np.int64)]
    masks = np.asarray(target_masks, dtype=bool)[np.asarray(train_indices, dtype=np.int64)]
    flat = actual.reshape(-1, TARGET_COUNT)
    complete = np.all(masks.reshape(-1, TARGET_COUNT), axis=1) & np.all(
        np.isfinite(flat), axis=1
    )
    if not np.any(complete):
        raise ValueError("training layer has no complete hours for frozen thresholds")
    return np.quantile(flat[complete], 0.05, axis=0, method="linear")


def threshold_hit_score(
    actual: np.ndarray, prediction: np.ndarray, frozen_thresholds: np.ndarray
) -> dict[str, object]:
    y = np.asarray(actual, dtype=np.float64).reshape(-1, TARGET_COUNT)
    p = np.asarray(prediction, dtype=np.float64).reshape(-1, TARGET_COUNT)
    thresholds = np.asarray(frozen_thresholds, dtype=np.float64)
    complete = np.all(np.isfinite(y), axis=1) & np.all(np.isfinite(p), axis=1)
    selected = complete & np.all(y >= thresholds[None, :], axis=1)
    if not np.any(selected) or np.any(y[selected] <= 0.0):
        raise ValueError("frozen threshold filter retained no strictly-positive hours")
    error = np.mean(np.abs(y[selected] - p[selected]) / y[selected], axis=1)
    rates = [float(np.mean(error < threshold)) for threshold in STRICT_THRESHOLDS]
    return {
        "name": "threshold_hit_score",
        "not_auc": True,
        "n_hours": int(np.sum(selected)),
        "frozen_training_thresholds": thresholds.tolist(),
        "mean_mape": float(np.mean(error)),
        "rates": rates,
        "score": float(np.mean(rates)),
    }


def forecast_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    mase_scales: np.ndarray,
    cells: np.ndarray,
) -> dict[str, object]:
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    scales = np.asarray(mase_scales, dtype=np.float64)
    cells = np.asarray(cells).astype(str)
    if y.shape != p.shape or y.shape[1:] != (FORECAST_HOURS, TARGET_COUNT):
        raise ValueError("forecast metric arrays are misaligned")
    indicator_rows: list[dict[str, float | int]] = []
    for metric in range(TARGET_COUNT):
        valid = np.isfinite(y[..., metric]) & np.isfinite(p[..., metric])
        yy = y[..., metric][valid]
        pp = p[..., metric][valid]
        absolute = np.abs(yy - pp)
        scale_grid = np.broadcast_to(scales[:, None, metric], y[..., metric].shape)
        scale_valid = valid & np.isfinite(scale_grid) & (scale_grid > 1e-12)
        mase = np.mean(
            np.abs(y[..., metric][scale_valid] - p[..., metric][scale_valid])
            / scale_grid[scale_valid]
        )
        indicator_rows.append(
            {
                "indicator": METRIC_NAMES[metric],
                "n": int(np.sum(valid)),
                "mae": float(np.mean(absolute)),
                "rmse": float(np.sqrt(np.mean((yy - pp) ** 2))),
                "wape": float(np.sum(absolute) / max(np.sum(np.abs(yy)), 1e-12)),
                "smape": float(
                    np.mean(2.0 * absolute / np.maximum(np.abs(yy) + np.abs(pp), 1e-12))
                ),
                "mase": float(mase),
            }
        )
    valid_all = np.isfinite(y) & np.isfinite(p)
    pooled_abs = np.abs(y[valid_all] - p[valid_all])
    pooled_wape = float(
        np.sum(pooled_abs) / max(np.sum(np.abs(y[valid_all])), 1e-12)
    )
    cell_wapes: list[float] = []
    for cell in sorted(set(cells.tolist())):
        selected = cells == cell
        valid = np.isfinite(y[selected]) & np.isfinite(p[selected])
        numerator = np.sum(np.abs(y[selected][valid] - p[selected][valid]))
        denominator = np.sum(np.abs(y[selected][valid]))
        if denominator > 0.0:
            cell_wapes.append(float(numerator / denominator))
    horizon_wape: list[float] = []
    for horizon in range(FORECAST_HOURS):
        valid = np.isfinite(y[:, horizon, :]) & np.isfinite(p[:, horizon, :])
        numerator = np.sum(np.abs(y[:, horizon, :][valid] - p[:, horizon, :][valid]))
        denominator = np.sum(np.abs(y[:, horizon, :][valid]))
        horizon_wape.append(float(numerator / max(denominator, 1e-12)))
    return {
        "per_indicator": indicator_rows,
        "macro_indicator": {
            key: float(np.mean([float(row[key]) for row in indicator_rows]))
            for key in ("mae", "rmse", "wape", "smape", "mase")
        },
        "pooled_wape": pooled_wape,
        "macro_cell_wape": float(np.mean(cell_wapes)),
        "median_cell_wape": float(np.median(cell_wapes)),
        "cell_count": len(cell_wapes),
        "per_horizon_wape": horizon_wape,
    }


def cell_cluster_bootstrap_wape_delta(
    actual: np.ndarray,
    proposed: np.ndarray,
    baseline: np.ndarray,
    cells: np.ndarray,
    *,
    replicates: int = 5000,
    seed: int = 42,
) -> dict[str, object]:
    """Paired cell-cluster interval for macro-over-indicator WAPE.

    Each replicate retains every date and horizon for a sampled cell, computes
    one ratio-of-sums WAPE per indicator, and averages the four indicator
    WAPEs. The returned point estimate therefore exactly matches the primary
    WAPE estimand reported by forecast_metrics.
    """
    y = np.asarray(actual, dtype=np.float64)
    a = np.asarray(proposed, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    cells = np.asarray(cells).astype(str)
    if y.shape != a.shape or y.shape != b.shape:
        raise ValueError("bootstrap arrays must have identical shapes")
    if y.shape[1:] != (FORECAST_HOURS, TARGET_COUNT) or len(cells) != len(y):
        raise ValueError("bootstrap arrays do not match the forecasting contract")
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    unique = np.asarray(sorted(set(cells.tolist())))
    num_a = np.zeros((len(unique), TARGET_COUNT), dtype=np.float64)
    num_b = np.zeros_like(num_a)
    denom = np.zeros_like(num_a)
    for cell_index, cell in enumerate(unique):
        selected = cells == cell
        for metric in range(TARGET_COUNT):
            valid = (
                np.isfinite(y[selected, :, metric])
                & np.isfinite(a[selected, :, metric])
                & np.isfinite(b[selected, :, metric])
            )
            yy = y[selected, :, metric][valid]
            aa = a[selected, :, metric][valid]
            bb = b[selected, :, metric][valid]
            denom[cell_index, metric] = np.sum(np.abs(yy))
            num_a[cell_index, metric] = np.sum(np.abs(yy - aa))
            num_b[cell_index, metric] = np.sum(np.abs(yy - bb))

    eligible = np.any(denom > 0.0, axis=1)
    num_a, num_b, denom = num_a[eligible], num_b[eligible], denom[eligible]
    total_denominator = np.sum(denom, axis=0)
    if np.any(total_denominator <= 0.0):
        raise ValueError("every indicator needs a positive bootstrap denominator")
    proposed_indicator = np.sum(num_a, axis=0) / total_denominator
    baseline_indicator = np.sum(num_b, axis=0) / total_denominator
    proposed_macro = float(np.mean(proposed_indicator))
    baseline_macro = float(np.mean(baseline_indicator))
    point = proposed_macro - baseline_macro

    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sample = rng.integers(0, len(denom), size=len(denom))
        sampled_denominator = np.sum(denom[sample], axis=0)
        sampled_denominator = np.maximum(sampled_denominator, 1e-12)
        indicator_delta = (
            np.sum(num_a[sample], axis=0) / sampled_denominator
            - np.sum(num_b[sample], axis=0) / sampled_denominator
        )
        draws[replicate] = float(np.mean(indicator_delta))
    return {
        "estimand": "macro_over_indicator_wape",
        "replicates": replicates,
        "seed": seed,
        "clusters": int(len(denom)),
        "point_proposed_macro_wape": proposed_macro,
        "point_baseline_macro_wape": baseline_macro,
        "point_proposed_indicator_wape": proposed_indicator.tolist(),
        "point_baseline_indicator_wape": baseline_indicator.tolist(),
        "delta_proposed_minus_baseline": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "probability_delta_below_zero": float(np.mean(draws < 0.0)),
    }
