"""Batched POMO for HPCVRP: all start rollouts run in parallel as tensor ops on GPU.

This replaces the slow per-start Python env loop with a vectorised construction, enabling the
heavy training (thousands of updates) needed to scale neural CO to 150-330 nodes. The batched
transition matches HPCVRPEnv semantics (append / close@>=Q_S / finish-always), and greedy decode
is cross-checked against the Python env.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from pdcrl.agents.am_policy import AMPolicy, encode_instance, node_features
from pdcrl.data.loader import transition_matrices
from pdcrl.envs.hpcvrp_env import HPCVRPEnv
from pdcrl.rewards.costs import full_weights
from pdcrl.utils.seeding import seed_everything


@dataclass
class BatchPOMOConfig:
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 2
    clip: float = 10.0
    lr: float = 3e-4
    lr_final: float | None = 1e-5  # cosine-anneal to this over num_updates; None -> constant lr
    entropy_coef: float = 0.01
    grad_clip: float = 1.0
    device: str = "cpu"
    rich_ctx: bool = False


class _Tensors:
    """Precomputed per-instance tensors on device."""
    def __init__(self, instance, profile, device):
        from pdcrl.agents.am_policy import edge_scales
        from pdcrl.data.loader import transition_component_matrices

        Dw, Dg, Dh, PE = transition_component_matrices(instance, profile)
        self._Dw = torch.as_tensor(Dw, dtype=torch.float32, device=device)
        self._Dg = torch.as_tensor(Dg, dtype=torch.float32, device=device)
        self._Dh = torch.as_tensor(Dh, dtype=torch.float32, device=device)
        self._Dfull = self._Dw + self._Dg + self._Dh
        self.D = self._Dfull  # reward-side D; defaults to the full penalty (backward compatible)
        self.PE = torch.as_tensor(PE, dtype=torch.float32, device=device)
        d_ref, pe_ref = edge_scales(profile)
        self.D_norm = self._Dfull / d_ref  # network-facing normalisation: always full-scale
        self.PE_norm = self.PE / pe_ref
        self.Q = torch.as_tensor([s.rolling_length_m for s in instance.slabs], dtype=torch.float32, device=device)
        self.prize = torch.as_tensor(
            [profile.prize_base * s.priority for s in instance.slabs], dtype=torch.float32, device=device)
        self.feats = node_features(instance, profile).to(device)
        self.n = instance.num_slabs
        self.Qs = float(profile.capacity_min_m)
        self.Ql = float(profile.capacity_max_m)
        self.Croll = float(profile.roll_change_cost)

    def set_s1_weights(self, w_width: float, w_gauge: float, w_hardness: float) -> None:
        """Reweight the reward-side transition matrix by process component (process_staged
        curriculum). Only ``self.D`` changes; ``self.D_norm`` (network input) stays full-scale."""
        self.D = w_width * self._Dw + w_gauge * self._Dg + w_hardness * self._Dh

    def reset_s1(self) -> None:
        """Restore the reward-side D to the full penalty (used before every true-objective eval)."""
        self.D = self._Dfull


def batched_rollout(policy, T, starts, weights, greedy=False):
    """Run len(starts) parallel rollouts. Returns (logp_sum(B,), ent_sum(B,), cost(B,), ret(B,)).

    ``cost`` is the true objective (w0*S1 + w1*S2 + w2*S3). ``ret`` is a *dense* reward,
    ``w1*prize[j] - (w0*d + w2*pe)`` on each append (no terminal prize) — mathematically
    ``w1*TotalPrize - cost``, so maximising ret == minimising cost, but with an immediate
    per-append signal that fixes the finish-early collapse at scale."""
    dev = T.D.device
    n = T.n
    B = len(starts)
    w0, w1, w2, w3 = (float(x) for x in full_weights(weights))
    enc = policy.encode(T.feats, T.D_norm, T.PE_norm, T.prize, T.Qs, T.Ql)
    starts = torch.as_tensor(starts, dtype=torch.long, device=dev)

    scheduled = torch.zeros(B, n, dtype=torch.bool, device=dev)
    scheduled[torch.arange(B, device=dev), starts] = True
    unit_mask = scheduled.clone()
    cur_len = T.Q[starts].clone()
    # forced start opens a unit: charge the roll change three ways (cost / ret / cur_wcost)
    open_c = w3 * T.Croll
    cur_wcost = torch.full((B,), open_c, device=dev)  # weighted cost inside the open unit
    last = starts.clone()
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    cost = torch.full((B,), open_c, device=dev)
    ret = torch.full((B,), -open_c, device=dev)
    logp_sum = torch.zeros(B, device=dev)
    ent_sum = torch.zeros(B, device=dev)
    arange = torch.arange(B, device=dev)

    for _ in range(3 * n + 2):
        if bool(done.all()):
            break
        append_ok = (~scheduled) & ((cur_len.unsqueeze(1) + T.Q.unsqueeze(0)) <= T.Ql)  # (B,n)
        close_ok = (last >= 0) & (cur_len >= T.Qs)                                       # (B,)
        # finish only when the current unit is empty or valid (>= Q_S), or when stuck (no append
        # possible) -> matches HPCVRPEnv's finish action + auto force-finish. This prevents the
        # trivial "finish immediately" escape that a finish-always mask would allow.
        finish_ok = (last < 0) | (cur_len >= T.Qs) | (~append_ok.any(dim=1))
        mask = torch.cat([append_ok, close_ok.unsqueeze(1), finish_ok.unsqueeze(1)], dim=1)
        # done rollouts: only finish, ignored in accumulation
        mask[done] = False
        mask[done, n + 1] = True

        logits = policy.decode_batch(enc, scheduled, unit_mask, last, cur_len, mask)
        dist = Categorical(logits=logits)
        a = logits.argmax(-1) if greedy else dist.sample()
        active = ~done
        logp_sum = logp_sum + dist.log_prob(a) * active
        ent_sum = ent_sum + dist.entropy() * active

        is_finish = (a == n + 1) & active
        is_close = (a == n) & active
        is_append = (a < n) & active
        j = a.clamp(max=n - 1)

        # append cost (transition only if in an open unit)
        has_last = (last >= 0) & is_append
        dtrans = T.D[last.clamp(min=0), j]
        petrans = T.PE[last.clamp(min=0), j]
        step_c = torch.where(has_last, w0 * dtrans + w2 * petrans,
                             torch.full_like(dtrans, open_c))
        cost = cost + torch.where(is_append, step_c, torch.zeros_like(step_c))
        # dense reward: collect w1*prize on append, pay transition/energy
        step_ret = w1 * T.prize[j] - step_c
        ret = ret + torch.where(is_append, step_ret, torch.zeros_like(step_ret))
        rows = arange[is_append]
        scheduled[rows, j[rows]] = True
        unit_mask[rows, j[rows]] = True
        cur_len = torch.where(is_append, cur_len + T.Q[j], cur_len)
        cur_wcost = torch.where(is_append, cur_wcost + step_c, cur_wcost)
        last = torch.where(is_append, j, last)
        # close
        unit_mask = unit_mask & ~is_close.unsqueeze(1)
        cur_len = torch.where(is_close, torch.zeros_like(cur_len), cur_len)
        cur_wcost = torch.where(is_close, torch.zeros_like(cur_wcost), cur_wcost)
        last = torch.where(is_close, torch.full_like(last, -1), last)
        # finish. A short open unit (< Q_S) is *released* to match HPCVRPEnv: its slabs go back
        # to unscheduled (prize charged below) and its internal transitions are refunded.
        release = is_finish & (last >= 0) & (cur_len < T.Qs)
        cost = torch.where(release, cost - cur_wcost, cost)
        unit_prize = (T.prize.unsqueeze(0) * unit_mask.float()).sum(1)
        ret = torch.where(release, ret + cur_wcost - w1 * unit_prize, ret)
        scheduled = scheduled & ~(unit_mask & release.unsqueeze(1))
        # prize for everything unscheduled (incl. just-released slabs)
        prize_left = (T.prize.unsqueeze(0) * (~scheduled).float()).sum(1)
        cost = torch.where(is_finish, cost + w1 * prize_left, cost)
        done = done | is_finish

    # force-finish any leftover (should be rare given finish-always)
    if not bool(done.all()):
        prize_left = (T.prize.unsqueeze(0) * (~scheduled).float()).sum(1)
        cost = torch.where(~done, cost + w1 * prize_left, cost)
    return logp_sum, ent_sum, cost, ret


def train_pomo_batched(instance, profile, cfg: BatchPOMOConfig, num_updates: int,
                       weights=(1, 1, 1, 1), num_starts=None, seed: int = 0, init_state=None):
    """Train (or, with ``init_state``, fine-tune from existing weights) on one instance."""
    seed_everything(seed)
    T = _Tensors(instance, profile, cfg.device)
    policy = AMPolicy(cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.clip, rich_ctx=getattr(cfg, "rich_ctx", False)).to(cfg.device)
    if init_state is not None:
        policy.load_state_dict(init_state)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    sched = None if cfg.lr_final is None else \
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_updates, eta_min=cfg.lr_final)
    n = T.n
    starts = list(range(n)) if num_starts is None else list(range(min(num_starts, n)))
    hist = []
    for _ in range(num_updates):
        logp, ent, cost, ret = batched_rollout(policy, T, starts, weights, greedy=False)
        # maximise dense return (REINFORCE): advantage = ret - shared baseline (mean over starts)
        adv = (ret - ret.mean()) / (ret.std() + 1e-6)
        loss = -(adv.detach() * logp).mean() - cfg.entropy_coef * ent.mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        opt.step()
        if sched is not None:
            sched.step()
        hist.append(float(cost.mean()))
    return policy, hist


def greedy_decode_batched(policy, instance, profile, weights=(1, 1, 1, 1)):
    """Batched greedy (argmax) multi-start; return best objective and the reconstructed Schedule."""
    device = next(policy.parameters()).device
    T = _Tensors(instance, profile, device)
    n = T.n
    with torch.no_grad():
        _, _, cost, _ = batched_rollout(policy, T, list(range(n)), weights, greedy=True)
    best_start = int(cost.argmin())
    # reconstruct the best rollout deterministically in the Python env
    env = HPCVRPEnv(instance, profile, objective_weights=full_weights(weights))
    obs, _ = env.reset(start_slab=best_start)
    terminated = False
    info = {}
    with torch.no_grad():
        enc = encode_instance(policy, instance, profile)
        while not terminated:
            logits = policy.action_logits(enc, obs)
            obs, _, terminated, _, info = env.step(int(logits.argmax()))
    return info["cost"]["objective_value"], info["schedule"]
