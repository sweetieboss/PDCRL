"""Edge-featured graph-attention policy for the HPCVRP construction MDP.

Encoder: graph attention over slabs where the transition-cost matrices D/PE bias every
attention head (Graphormer-style edge features) — the routing "distances" are first-class
graph structure, not discarded. Decoder: a *dynamic* context recomputed each step from the
actual construction state (masked mean over unscheduled slabs, masked mean over the open
unit, last-slab embedding, progress scalars), one glimpse attention over the unscheduled
set, then a pointer whose slab logits combine embedding compatibility with the direct edge
cost D/PE[last, j] — so the policy can represent the greedy heuristic exactly and improve
on it. Actions: append slab j (0..n-1), close unit (n), finish (n+1); feasibility masking
is supplied by the environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIM = 6


def power_span(profile) -> tuple[float, float]:
    """Profile-level power-per-ton range (matches the generator's grade+gauge model)."""
    lo = 100.0 + 2.0 * profile.gauge_min_mm
    hi = 100.0 + 15.0 * (profile.num_grades - 1) + 2.0 * profile.gauge_max_mm
    return lo, hi


def edge_scales(profile) -> tuple[float, float]:
    """Profile-level reference magnitudes for D and PE, so edge normalisation is
    instance-independent (a policy trained on small instances sees the same edge scale on
    large ones — required for clean cross-scale curriculum transfer)."""
    d_ref = (profile.width_up_penalty * (profile.width_max_mm - profile.width_min_mm)
             + profile.gauge_penalty * (profile.gauge_max_mm - profile.gauge_min_mm)
             + profile.hardness_penalty * (profile.hardness_max - profile.hardness_min))
    lo, hi = power_span(profile)
    pe_ref = profile.energy_penalty * (hi - lo)
    return float(d_ref), float(pe_ref)


def node_features(instance, profile) -> torch.Tensor:
    """Normalised per-slab features (n, 6): width, gauge, hardness, rolling-length, priority,
    power. All scales come from the *profile* (never from instance statistics), so the same
    physical slab has identical features in any instance — cross-scale transfer stays clean."""
    n = instance.num_slabs
    f = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    wspan = max(1e-9, profile.width_max_mm - profile.width_min_mm)
    gspan = max(1e-9, profile.gauge_max_mm - profile.gauge_min_mm)
    pe_lo, pe_hi = power_span(profile)
    for i, s in enumerate(instance.slabs):
        f[i] = [
            (s.width_mm - profile.width_min_mm) / wspan,
            (s.gauge_mm - profile.gauge_min_mm) / gspan,
            (s.hardness - profile.hardness_min) / max(1, profile.hardness_max - profile.hardness_min),
            s.rolling_length_m / max(1e-9, profile.capacity_max_m),  # fraction of unit capacity
            (s.priority - 1) / max(1, profile.priority_max - 1),
            (s.power_per_ton - pe_lo) / max(1e-9, pe_hi - pe_lo),
        ]
    return torch.from_numpy(f)


@dataclass
class Encoded:
    """Static per-instance quantities computed once per encode; reused every decode step."""
    H: torch.Tensor        # (n, d) node embeddings
    K: torch.Tensor        # (n, d) pointer keys
    Kg: torch.Tensor       # (h, n, dh) glimpse keys (projected ONCE — projecting per step
    Vg: torch.Tensor       # (h, n, dh) glimpse values   blows up rollout memory/time)
    D_norm: torch.Tensor   # (n, n) normalised transition penalty
    PE_norm: torch.Tensor  # (n, n) normalised energy-jump penalty
    prize_norm: torch.Tensor  # (n,) prize / total prize
    w_feat: torch.Tensor   # (n,) normalised width (feature col 0) — rich context coordination scalars
    Qs: float
    Ql: float


class EdgeBiasAttentionLayer(nn.Module):
    """Multi-head self-attention whose logits are biased per head by edge features (D/PE)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, edge_bias: torch.Tensor) -> torch.Tensor:
        """x (n, d); edge_bias (n_heads, n, n) added to the attention logits of every head."""
        n = x.shape[0]
        q, k, v = self.qkv(x).view(n, 3, self.h, self.dh).permute(1, 2, 0, 3)  # each (h, n, dh)
        att = torch.softmax(q @ k.transpose(-1, -2) / self.dh ** 0.5 + edge_bias, dim=-1)
        o = (att @ v).transpose(0, 1).reshape(n, self.h * self.dh)
        x = self.norm1(x + self.out(o))
        return self.norm2(x + self.ff(x))


class AMPolicy(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 8, n_layers: int = 2, clip: float = 10.0,
                 rich_ctx: bool = False):
        super().__init__()
        self.d = d_model
        self.clip = clip
        self.rich_ctx = rich_ctx
        self.n_scalars = 8 if rich_ctx else 4    # +4 process-coordination scalars
        self.embed = nn.Linear(FEATURE_DIM, d_model)
        self.edge_proj = nn.Linear(2, n_heads)
        self.layers = nn.ModuleList(EdgeBiasAttentionLayer(d_model, n_heads) for _ in range(n_layers))
        self.q_proj = nn.Linear(3 * d_model + self.n_scalars, d_model)
        self.h_glimpse = n_heads
        self.g_q = nn.Linear(d_model, d_model)
        self.g_kv = nn.Linear(d_model, 2 * d_model)
        self.g_out = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.close_head = nn.Linear(d_model, 1)
        self.finish_head = nn.Linear(d_model, 1)
        # learned (positive) weights on the direct edge-cost pointer bias
        self.w_d = nn.Parameter(torch.tensor(1.0))
        self.w_pe = nn.Parameter(torch.tensor(1.0))

    # ---- static, once per instance ----
    def encode(self, feats, D_norm, PE_norm, prize, Qs, Ql) -> Encoded:
        edge_bias = self.edge_proj(torch.stack([D_norm, PE_norm], dim=-1)).permute(2, 0, 1)  # (h,n,n)
        h = self.embed(feats)
        for layer in self.layers:
            h = layer(h, edge_bias)
        n = h.shape[0]
        dh = self.d // self.h_glimpse
        kg, vg = self.g_kv(h).view(n, 2, self.h_glimpse, dh).permute(1, 2, 0, 3)  # each (h,n,dh)
        return Encoded(H=h, K=self.k_proj(h), Kg=kg, Vg=vg, D_norm=D_norm, PE_norm=PE_norm,
                       prize_norm=prize / (prize.sum() + 1e-9), w_feat=feats[:, 0].contiguous(),
                       Qs=float(Qs), Ql=float(Ql))

    # ---- dynamic, every step ----
    def decode_batch(self, enc: Encoded, scheduled, unit_mask, last, cur_len, action_mask) -> torch.Tensor:
        """State -> masked logits (B, n+2).

        scheduled (B,n) bool; unit_mask (B,n) bool (open unit membership); last (B,) long
        (-1 = empty unit); cur_len (B,) float; action_mask (B,n+2) bool.
        """
        H, K = enc.H, enc.K
        B = scheduled.shape[0]
        unsched = ~scheduled

        def masked_mean(m):
            cnt = m.float().sum(1, keepdim=True).clamp(min=1.0)
            return (m.float() @ H) / cnt   # all-False row -> zero vector

        remain = (enc.Ql - cur_len.float()) / enc.Ql
        qs_prog = (cur_len.float() / enc.Qs).clamp(max=1.0)
        frac_sched = scheduled.float().mean(1)
        prize_left = (enc.prize_norm.unsqueeze(0) * unsched.float()).sum(1)
        cols = [remain, qs_prog, frac_sched, prize_left]
        if self.rich_ctx:
            # coordination scalars over the open unit / remaining pool (all in [0,1], profile-
            # normalised widths -> transfer-clean): current-unit width ceiling (governs coffin),
            # remaining-pool width range, and the fraction of remaining slabs addable without a
            # width-increase penalty.
            w = enc.w_feat.unsqueeze(0)                                   # (1,n)
            neg = torch.full_like(w, -1.0); pos = torch.full_like(w, 2.0)
            unit_wmax = torch.where(unit_mask, w, neg).max(1).values.clamp(min=0.0)   # 0 if empty
            pool_wmax = torch.where(unsched, w, neg).max(1).values.clamp(min=0.0)
            pool_wmin = torch.where(unsched, w, pos).min(1).values.clamp(max=1.0)
            addable = (unsched & (w <= unit_wmax.unsqueeze(1) + 1e-6)).float().sum(1)
            feasible_frac = addable / unsched.float().sum(1).clamp(min=1.0)
            cols += [unit_wmax, pool_wmax, pool_wmin, feasible_frac]
        scalars = torch.stack(cols, dim=1)  # (B, n_scalars)

        has_last = (last >= 0).unsqueeze(1)
        last_emb = torch.where(has_last, H[last.clamp(min=0)], torch.zeros(1, self.d, device=H.device, dtype=H.dtype))
        q = self.q_proj(torch.cat([masked_mean(unsched), masked_mean(unit_mask), last_emb, scalars], dim=-1))

        # one glimpse over the unscheduled set (keys/values precomputed in encode; per-step
        # work is just the query projection + one masked attention — no K/V reprojection)
        dh = self.d // self.h_glimpse
        kpm = scheduled & unsched.any(dim=1, keepdim=True)   # avoid all-masked NaN rows
        qg = self.g_q(q).view(B, self.h_glimpse, 1, dh)
        att = (qg * enc.Kg.unsqueeze(0)).sum(-1) / dh ** 0.5          # (B,h,n)
        att = att.masked_fill(kpm.unsqueeze(1), float("-inf")).softmax(dim=-1)
        ctx = (att.unsqueeze(-1) * enc.Vg.unsqueeze(0)).sum(2)        # (B,h,dh)
        q = q + self.g_out(ctx.reshape(B, self.d))

        # pointer: embedding compatibility minus the direct edge cost from the last slab
        compat = (q @ K.t()) / self.d ** 0.5                                   # (B,n)
        edge = F.softplus(self.w_d) * enc.D_norm[last.clamp(min=0)] \
            + F.softplus(self.w_pe) * enc.PE_norm[last.clamp(min=0)]           # (B,n)
        slab = self.clip * torch.tanh(compat - has_last.float() * edge)
        close = self.clip * torch.tanh(self.close_head(q))                     # (B,1)
        finish = self.clip * torch.tanh(self.finish_head(q))                   # (B,1)
        logits = torch.cat([slab, close, finish], dim=-1)
        return logits.masked_fill(~action_mask, -1e9)

    def action_logits(self, enc: Encoded, obs) -> torch.Tensor:
        """Single-state wrapper over decode_batch for the Python env (obs dict from HPCVRPEnv)."""
        dev = enc.H.device
        return self.decode_batch(
            enc,
            torch.as_tensor(np.asarray(obs["scheduled"], dtype=bool), device=dev).unsqueeze(0),
            torch.as_tensor(np.asarray(obs["unit_mask"], dtype=bool), device=dev).unsqueeze(0),
            torch.tensor([obs["last"] if obs["last"] is not None else -1], dtype=torch.long, device=dev),
            torch.tensor([float(obs["cur_len"])], device=dev),
            torch.as_tensor(np.asarray(obs["action_mask"], dtype=bool), device=dev).unsqueeze(0),
        ).squeeze(0)


def encode_instance(policy: AMPolicy, instance, profile) -> Encoded:
    """Build the encoder inputs (features, normalised D/PE, prizes) on the policy's device."""
    from pdcrl.data.loader import transition_matrices

    dev = next(policy.parameters()).device
    D, PE = transition_matrices(instance, profile)
    D = torch.as_tensor(D, dtype=torch.float32, device=dev)
    PE = torch.as_tensor(PE, dtype=torch.float32, device=dev)
    d_ref, pe_ref = edge_scales(profile)
    prize = torch.as_tensor([profile.prize_base * s.priority for s in instance.slabs],
                            dtype=torch.float32, device=dev)
    return policy.encode(node_features(instance, profile).to(dev),
                         D / d_ref, PE / pe_ref,
                         prize, profile.capacity_min_m, profile.capacity_max_m)
