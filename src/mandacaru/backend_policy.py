"""Validate obsolete backend environment settings at the native ABI boundary."""
import os


def backend_preference():
    value = os.environ.get("MANDACARU_BACKEND", "rust").strip().lower() or "rust"
    if value != "rust":
        raise ValueError("Mandacaru requires Rust; remove MANDACARU_BACKEND")
    return value
