#!/usr/bin/python3
"""Fail-closed decoder for the version-pinned AD102 GMMU v2 PTE contract.

Ported verbatim from the REMU gpu_va_pa_mapping observer
(ada_pte_decoder.py, frozen 2026-09-08). The decoder refuses to guess:
unknown entry sizes, out-of-range words, or non-zero high words on
16-byte entries never yield a local-PA classification.
"""

from __future__ import annotations

from typing import Any


def _field(value: int, lsb: int, width: int) -> int:
    return (value >> lsb) & ((1 << width) - 1)


def decode_pte(raw_pte: int, pte_size: int, contract: dict[str, Any], raw_pte_hi: int = 0) -> dict[str, Any]:
    fmt = contract["pte_format"]
    supported_sizes = fmt["supported_entry_sizes_bytes"]
    if pte_size not in supported_sizes:
        raise ValueError(f"unsupported entry size {pte_size}; expected one of {supported_sizes}")
    if raw_pte < 0 or raw_pte >= (1 << fmt["pte_payload_bits"]):
        raise ValueError("low PTE word is outside the declared width")
    if raw_pte_hi < 0 or raw_pte_hi >= (1 << 64):
        raise ValueError("high PTE word is outside 64 bits")
    high_word_zero = pte_size == 8 or raw_pte_hi == 0

    aperture_code = _field(raw_pte, fmt["aperture_lsb"], fmt["aperture_width"])
    aperture = fmt["aperture_labels"].get(str(aperture_code), "UNKNOWN")
    valid = bool(_field(raw_pte, fmt["valid_lsb"], 1))
    if aperture in ("VIDEO", "PEER"):
        address_field = _field(raw_pte, fmt["video_address_lsb"], fmt["video_address_width"])
    elif aperture in ("SYS_COHERENT", "SYS_NONCOHERENT"):
        address_field = _field(raw_pte, fmt["system_address_lsb"], fmt["system_address_width"])
    else:
        address_field = 0

    physical_base = address_field << fmt["address_shift"] if valid and aperture != "UNKNOWN" and high_word_zero else None
    return {
        "raw_pte_lo": f"0x{raw_pte:016x}",
        "raw_pte_hi": f"0x{raw_pte_hi:016x}" if pte_size == 16 else "",
        "raw_entry": f"0x{raw_pte_hi:016x}{raw_pte:016x}" if pte_size == 16 else f"0x{raw_pte:016x}",
        "entry_size": pte_size,
        "high_word_zero": high_word_zero,
        "valid": valid,
        "aperture_code": aperture_code,
        "aperture": aperture,
        "kind": _field(raw_pte, fmt["kind_lsb"], fmt["kind_width"]),
        "physical_base": None if physical_base is None else f"0x{physical_base:x}",
        "gpu_local_pa": aperture == "VIDEO" and valid and high_word_zero,
    }


def format_uuid(raw: bytes) -> str:
    if len(raw) != 16:
        raise ValueError(f"GPU UUID must contain 16 bytes, got {len(raw)}")
    value = raw.hex()
    return f"{value[:8]}-{value[8:12]}-{value[12:16]}-{value[16:20]}-{value[20:]}"
