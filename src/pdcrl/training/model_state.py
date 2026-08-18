"""Deterministic checkpoint hashes for inference immutability audits."""

from __future__ import annotations

import hashlib


def state_dict_sha256(state_dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().to("cpu").contiguous()
        for value in (key, str(tensor.dtype), repr(tuple(tensor.shape))):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()

