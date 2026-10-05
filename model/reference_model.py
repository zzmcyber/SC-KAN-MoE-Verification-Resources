"""SC-KAN-MoE architecture reference with configurable model components."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    input_cont_dim: int
    n_climate_l1: int
    n_climate_l2: int
    n_climate_l3: int
    n_land_cover: int
    n_months: int
    n_seasons: int
    num_experts: int
    top_k: int
    context_embed_dim: int
    time_embed_dim: int
    dropout: float
    grid_size: int
    spline_order: int
    grid_min: float
    grid_max: float
    spline_init_std: float
    expert_hidden_dim: int
    regressor_hidden_dim: int
    use_context_embeddings: bool
    use_time_embeddings: bool

    def validate(self) -> None:
        positive = (
            "input_cont_dim", "num_experts", "top_k", "grid_size",
            "expert_hidden_dim", "regressor_hidden_dim",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.top_k < 2:
            raise ValueError("top_k must be at least 2 for selected-weight renormalization")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.spline_order < 0 or self.grid_min >= self.grid_max:
            raise ValueError("invalid spline grid")
        if self.spline_init_std < 0:
            raise ValueError("spline_init_std cannot be negative")
        if self.use_context_embeddings:
            for name in ("n_climate_l1", "n_climate_l2", "n_climate_l3", "n_land_cover", "context_embed_dim"):
                if getattr(self, name) <= 0:
                    raise ValueError(f"{name} must be positive when context embeddings are enabled")
        if self.use_time_embeddings:
            for name in ("n_months", "n_seasons", "time_embed_dim"):
                if getattr(self, name) <= 0:
                    raise ValueError(f"{name} must be positive when time embeddings are enabled")


class KANLinear(nn.Module):
    """Linear SiLU branch plus a trainable B-spline branch."""

    def __init__(self, in_features: int, out_features: int, config: ModelConfig):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = config.grid_size
        self.spline_order = config.spline_order
        step = (config.grid_max - config.grid_min) / config.grid_size
        grid = torch.arange(
            -config.spline_order,
            config.grid_size + config.spline_order + 1,
            dtype=torch.float32,
        ) * step + config.grid_min
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, config.grid_size + config.spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.ones(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(
            self.spline_weight,
            mean=0.0,
            std=config.spline_init_std / config.grid_size,
        )

    def b_splines(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f"expected [batch, {self.in_features}] continuous input")
        grid = self.grid
        expanded = x.unsqueeze(-1)
        basis = ((expanded >= grid[:, :-1]) & (expanded < grid[:, 1:])).to(x.dtype)
        for order in range(1, self.spline_order + 1):
            left_num = expanded - grid[:, : -(order + 1)]
            left_den = grid[:, order:-1] - grid[:, : -(order + 1)]
            right_num = grid[:, order + 1 :] - expanded
            right_den = grid[:, order + 1 :] - grid[:, 1:-order]
            basis = left_num / left_den.clamp_min(1e-12) * basis[:, :, :-1] + (
                right_num / right_den.clamp_min(1e-12) * basis[:, :, 1:]
            )
        return basis.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        base = F.linear(F.silu(x), self.base_weight)
        spline = F.linear(
            self.b_splines(x).reshape(x.shape[0], -1),
            (self.spline_weight * self.spline_scaler.unsqueeze(-1)).reshape(self.out_features, -1),
        )
        return base + spline


class GRKAN(nn.Module):
    def __init__(self, dim: int, config: ModelConfig):
        super().__init__()
        self.gate = nn.Sequential(
            KANLinear(dim, config.expert_hidden_dim, config),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            KANLinear(config.expert_hidden_dim, dim, config),
        )
        self.residual = KANLinear(dim, dim, config)

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.gate(x)


class KAMoE(nn.Module):
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        config.validate()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.gate = GRKAN(input_dim, config)
        self.gate_out = nn.Linear(input_dim, config.num_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(
                GRKAN(input_dim, config),
                nn.Linear(input_dim, config.expert_hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.expert_hidden_dim, input_dim),
            )
            for _ in range(config.num_experts)
        )

    def forward(self, x: Tensor, return_aux: bool = False):
        logits = self.gate_out(self.gate(x))
        weights = F.softmax(logits, dim=-1)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        selected_weights, selected_indices = torch.topk(weights, self.top_k, dim=-1)
        selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        selected_outputs = torch.gather(
            outputs, dim=1,
            index=selected_indices.unsqueeze(-1).expand(-1, -1, outputs.shape[-1]),
        )
        result = (selected_outputs * selected_weights.unsqueeze(-1)).sum(dim=1)
        if not return_aux:
            return result
        return result, {
            "gate_logits": logits,
            "gate_weights": weights,
            "topk_indices": selected_indices,
            "topk_weights": selected_weights,
        }


class HierarchicalClimateEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.context_embed_dim
        self.l1 = nn.Embedding(config.n_climate_l1, dim)
        self.l2 = nn.Embedding(config.n_climate_l2, dim)
        self.l3 = nn.Embedding(config.n_climate_l3, dim)
        self.project = nn.Linear(3 * dim, dim)

    def forward(self, l1: Tensor, l2: Tensor, l3: Tensor) -> Tensor:
        return self.project(torch.cat([self.l1(l1), self.l2(l2), self.l3(l3)], dim=1))


class ReferenceSoilMoistureModel(nn.Module):
    """Predict the transformed residual target."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        context_dim = 0
        if config.use_context_embeddings:
            self.climate = HierarchicalClimateEmbedding(config)
            self.land_cover = nn.Embedding(config.n_land_cover, config.context_embed_dim)
            context_dim = 2 * config.context_embed_dim
        time_dim = 0
        if config.use_time_embeddings:
            self.month = nn.Embedding(config.n_months, config.time_embed_dim)
            self.season = nn.Embedding(config.n_seasons, config.time_embed_dim)
            time_dim = 2 * config.time_embed_dim
        self.total_input_dim = config.input_cont_dim + context_dim + time_dim
        self.moe = KAMoE(self.total_input_dim, config)
        self.regressor = nn.Sequential(
            KANLinear(self.total_input_dim, config.regressor_hidden_dim, config),
            nn.SiLU(),
            KANLinear(config.regressor_hidden_dim, 1, config),
        )

    def forward(
        self, continuous: Tensor, climate_l1: Tensor | None = None,
        climate_l2: Tensor | None = None, climate_l3: Tensor | None = None,
        land_cover: Tensor | None = None, month: Tensor | None = None,
        season: Tensor | None = None, return_aux: bool = False,
    ):
        if continuous.ndim != 2 or continuous.shape[1] != self.config.input_cont_dim:
            raise ValueError("continuous input has the wrong shape")
        parts = [continuous]
        if self.config.use_context_embeddings:
            if any(value is None for value in (climate_l1, climate_l2, climate_l3, land_cover)):
                raise ValueError("climate and land-cover indices are required")
            parts.extend([
                self.climate(climate_l1, climate_l2, climate_l3),
                self.land_cover(land_cover),
            ])
        if self.config.use_time_embeddings:
            if month is None or season is None:
                raise ValueError("month and season indices are required")
            parts.extend([self.month(month), self.season(season)])
        combined = torch.cat(parts, dim=1)
        if return_aux:
            moe_output, aux = self.moe(combined, return_aux=True)
            return self.regressor(combined + moe_output), aux
        return self.regressor(combined + self.moe(combined))


def reconstruct_sm(sm_lag_1: Tensor, physical_residual: Tensor, clip_bounds: tuple[float, float] | None = None) -> Tensor:
    """Update matching samples after inverse transformation; retain the residual shape."""
    for values in (sm_lag_1, physical_residual):
        if values.ndim > 2 or (values.ndim == 2 and values.shape[1] != 1):
            raise ValueError("soil-moisture inputs must be scalars, vectors, or single-column tensors")
    if sm_lag_1.numel() != physical_residual.numel():
        raise ValueError("antecedent states and residuals must contain the same number of samples")
    result = sm_lag_1.reshape_as(physical_residual) + physical_residual
    if clip_bounds is not None:
        lower, upper = clip_bounds
        if lower >= upper:
            raise ValueError("clip_bounds must be increasing")
        result = result.clamp(lower, upper)
    return result
