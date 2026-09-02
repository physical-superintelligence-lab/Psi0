"""Attention backend selection for Qwen3-VL loading."""

from __future__ import annotations

import os


def qwen3vl_attn_implementation() -> str:
    """Return the attention backend for Qwen3-VL.

    An explicit ``PSI_ATTN_IMPLEMENTATION`` wins. Otherwise preserve Psi0's
    automatic flash-attn detection and fall back to SDPA.
    """

    override = os.environ.get("PSI_ATTN_IMPLEMENTATION")
    if override:
        return override
    try:
        from transformers.utils import is_flash_attn_2_available

        return "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    except Exception:
        return "sdpa"
