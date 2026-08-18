"""Problem-adapted reproduction of published POMO for the HPCVRP.

The implementation is independent from PDCRL's process-specific policy: it has a vanilla
attention encoder, no edge bias, no direct D/PE pointer bias and no rich process context. It uses
the published forced-multiple-start/shared-baseline POMO principle. Upstream source is not copied
because https://github.com/yd-kwon/POMO contains no license file.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn as nn

from pdcrl.agents.am_policy import FEATURE_DIM
from pdcrl.agents.pomo import POMOConfig, _pomo_update
from pdcrl.agents.pomo_batched import _Tensors, batched_rollout
from pdcrl.utils.seeding import seed_everything


PUBLISHED_POMO_PROTOCOL = "published-pomo-reproduction"
PUBLISHED_POMO_REPOSITORY = "https://github.com/yd-kwon/POMO"
PUBLISHED_POMO_COMMIT = "d7c3d6ea580499a53e874fe9e065f69e799a8551"


@dataclass
class PublishedPOMOEncoded:
    H: torch.Tensor
    K: torch.Tensor
    Kg: torch.Tensor
    Vg: torch.Tensor
    prize_norm: torch.Tensor
    Qs: float
    Ql: float


class _VanillaAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(values, values, values, need_weights=False)
        values = self.norm1(values + attended)
        return self.norm2(values + self.feed_forward(values))


class PublishedPOMOPolicy(nn.Module):
    def __init__(self, d_model=128, n_heads=8, n_layers=3, clip=10.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d = int(d_model)
        self.h = int(n_heads)
        self.dh = self.d // self.h
        self.clip = float(clip)
        self.embed = nn.Linear(FEATURE_DIM, self.d)
        self.layers = nn.ModuleList(
            _VanillaAttentionLayer(self.d, self.h) for _ in range(n_layers)
        )
        self.query = nn.Linear(3 * self.d + 4, self.d)
        self.glimpse_query = nn.Linear(self.d, self.d)
        self.glimpse_kv = nn.Linear(self.d, 2 * self.d)
        self.glimpse_out = nn.Linear(self.d, self.d)
        self.pointer_key = nn.Linear(self.d, self.d)
        self.close_key = nn.Parameter(torch.empty(self.d))
        self.finish_key = nn.Parameter(torch.empty(self.d))
        nn.init.uniform_(self.close_key, -self.d**-0.5, self.d**-0.5)
        nn.init.uniform_(self.finish_key, -self.d**-0.5, self.d**-0.5)

    def encode(self, feats, D_norm, PE_norm, prize, Qs, Ql):
        del D_norm, PE_norm
        values = self.embed(feats).unsqueeze(0)
        for layer in self.layers:
            values = layer(values)
        hidden = values.squeeze(0)
        n = hidden.shape[0]
        glimpse_k, glimpse_v = (
            self.glimpse_kv(hidden)
            .view(n, 2, self.h, self.dh)
            .permute(1, 2, 0, 3)
        )
        return PublishedPOMOEncoded(
            H=hidden,
            K=self.pointer_key(hidden),
            Kg=glimpse_k,
            Vg=glimpse_v,
            prize_norm=prize / (prize.sum() + 1e-9),
            Qs=float(Qs),
            Ql=float(Ql),
        )

    def decode_batch(self, encoded, scheduled, unit_mask, last, cur_len, action_mask):
        del unit_mask
        hidden = encoded.H
        unscheduled = ~scheduled
        batch = scheduled.shape[0]

        def masked_mean(mask):
            denominator = mask.float().sum(1, keepdim=True).clamp(min=1.0)
            return mask.float() @ hidden / denominator

        last_exists = (last >= 0).unsqueeze(1)
        last_embedding = torch.where(
            last_exists,
            hidden[last.clamp(min=0)],
            torch.zeros(1, self.d, device=hidden.device, dtype=hidden.dtype),
        )
        scalars = torch.stack(
            [
                (encoded.Ql - cur_len.float()) / encoded.Ql,
                (cur_len.float() / encoded.Qs).clamp(max=1.0),
                scheduled.float().mean(1),
                (encoded.prize_norm.unsqueeze(0) * unscheduled.float()).sum(1),
            ],
            dim=1,
        )
        query = self.query(
            torch.cat(
                [masked_mean(unscheduled), hidden.mean(0).expand(batch, -1), last_embedding, scalars],
                dim=1,
            )
        )
        avoid_all_masked = scheduled & unscheduled.any(dim=1, keepdim=True)
        glimpse_query = self.glimpse_query(query).view(batch, self.h, 1, self.dh)
        attention = (glimpse_query * encoded.Kg.unsqueeze(0)).sum(-1) / self.dh**0.5
        attention = attention.masked_fill(
            avoid_all_masked.unsqueeze(1), float("-inf")
        ).softmax(dim=-1)
        context = (attention.unsqueeze(-1) * encoded.Vg.unsqueeze(0)).sum(2)
        query = query + self.glimpse_out(context.reshape(batch, self.d))
        slab = query @ encoded.K.t() / self.d**0.5
        close = (query * self.close_key).sum(1, keepdim=True) / self.d**0.5
        finish = (query * self.finish_key).sum(1, keepdim=True) / self.d**0.5
        logits = self.clip * torch.tanh(torch.cat([slab, close, finish], dim=1))
        return logits.masked_fill(~action_mask, -1e9)

    def action_logits(self, encoded, observation):
        device = encoded.H.device
        last = observation["last"] if observation["last"] is not None else -1
        return self.decode_batch(
            encoded,
            torch.as_tensor(np.asarray(observation["scheduled"], dtype=bool), device=device).unsqueeze(0),
            torch.as_tensor(np.asarray(observation["unit_mask"], dtype=bool), device=device).unsqueeze(0),
            torch.tensor([last], dtype=torch.long, device=device),
            torch.tensor([float(observation["cur_len"])], device=device),
            torch.as_tensor(np.asarray(observation["action_mask"], dtype=bool), device=device).unsqueeze(0),
        ).squeeze(0)


def train_published_pomo_fixed_budget(
    instances,
    profile,
    cfg: POMOConfig,
    *,
    seed: int,
    total_seconds: float,
    validation_instances=(),
    validation_reference_costs=None,
    eval_every_s=480.0,
    weight_decay=1.0e-6,
):
    if not instances:
        raise ValueError("at least one training instance is required")
    if total_seconds <= 0.0:
        raise ValueError("total_seconds must be positive")
    if cfg.lr_final is not None:
        raise ValueError("published POMO reproduction requires lr_final=None (constant LR)")
    if abs(cfg.entropy_coef) > 1e-12:
        raise ValueError("published POMO reproduction uses REINFORCE without entropy regularization")
    seed_everything(seed)
    policy = PublishedPOMOPolicy(cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.clip).to(cfg.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    cache = {}
    counts = Counter()
    history = {
        "protocol_implementation": PUBLISHED_POMO_PROTOCOL,
        "upstream_repository": PUBLISHED_POMO_REPOSITORY,
        "upstream_commit": PUBLISHED_POMO_COMMIT,
        "upstream_license": "no-license-file-detected",
        "implementation_scope": "problem-adapted reproduction; upstream source not redistributed",
        "curriculum": False,
        "crossfade": False,
        "checkpoint_restore": False,
        "rollback": False,
        "lr_schedule": "constant",
        "learning_rate": float(cfg.lr),
        "weight_decay": float(weight_decay),
        "entropy_coef": float(cfg.entropy_coef),
        "objective_weights": [1.0, 1.0, 1.0, 1.0],
        "cost": [],
        "validation_curve": [],
        "instance_updates": {},
        "total_rollouts": 0,
    }

    def tensors_for(instance):
        if id(instance) not in cache:
            cache[id(instance)] = _Tensors(instance, profile, cfg.device)
        return cache[id(instance)]

    references = validation_reference_costs or {}

    def validation_median():
        values = []
        for instance in validation_instances:
            starts = list(range(min(64, instance.num_slabs)))
            with torch.no_grad():
                _, _, costs, _ = batched_rollout(
                    policy, tensors_for(instance), starts, (1, 1, 1, 1), greedy=True
                )
            reference = float(references.get(instance.name, 1.0))
            if reference <= 0.0:
                raise ValueError(f"non-positive validation reference for {instance.name}")
            values.append(float(costs.min()) / reference)
        return float(np.median(values)) if values else float("nan")

    started = time.time()
    next_evaluation = float(eval_every_s)
    first = True
    while first or time.time() - started < total_seconds:
        first = False
        instance = instances[int(rng.integers(len(instances)))]
        starts = (
            list(range(instance.num_slabs))
            if cfg.num_starts is None
            else list(rng.choice(instance.num_slabs, min(cfg.num_starts, instance.num_slabs), replace=False))
        )
        history["cost"].append(
            _pomo_update(policy, optimizer, tensors_for(instance), starts, (1, 1, 1, 1), cfg)
        )
        history["total_rollouts"] += len(starts)
        counts[instance.name] += 1
        elapsed = time.time() - started
        if validation_instances and elapsed >= next_evaluation:
            next_evaluation += eval_every_s
            history["validation_curve"].append([round(elapsed, 3), validation_median()])
    history["instance_updates"] = dict(sorted(counts.items()))
    history["training_seconds"] = time.time() - started
    return policy, history
