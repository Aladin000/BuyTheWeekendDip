"""Deterministic random number generation.

The project seed is ``20260901`` with NumPy ``Generator(PCG64)``, and the
NumPy version is recorded.

Each random procedure draws from a named stream. The name is turned into a
``SeedSequence`` spawn key with SHA-256, so the same name always gives the same
draws and two names never share a sequence. Python's ``hash`` is not used
because it changes between interpreter runs.

This is the only source of randomness for results, so nothing here reads the
clock or system entropy. The ``uuid4`` in
:func:`src.artifacts.new_attempt_token` only labels run attempts and never
reaches a result.
"""

from __future__ import annotations

import hashlib

import numpy as np

from src.config import ConfigError, ConfigNode, load_config

_SPAWN_KEY_BITS = 64


def numpy_version() -> str:
    """NumPy version recorded with stochastic results."""
    return np.__version__


def stream_key(stream: str) -> int:
    """Stable non-negative spawn key for a named stream."""
    if not isinstance(stream, str) or not stream:
        raise ConfigError("stream name must be a non-empty string")
    digest = hashlib.sha256(stream.encode("utf-8")).digest()
    return int.from_bytes(digest[: _SPAWN_KEY_BITS // 8], "big")


def seed_sequence(stream: str, config: ConfigNode | None = None) -> np.random.SeedSequence:
    """``SeedSequence`` for a named stream under the locked project seed."""
    cfg = config if config is not None else load_config()
    if cfg.reproducibility.bit_generator != "PCG64":
        raise ConfigError(
            f"reproducibility.bit_generator {cfg.reproducibility.bit_generator!r} is not the "
            "locked PCG64 bit generator"
        )
    return np.random.SeedSequence(
        entropy=cfg.reproducibility.seed, spawn_key=(stream_key(stream),)
    )


def make_rng(stream: str, config: ConfigNode | None = None) -> np.random.Generator:
    """Return a fresh ``Generator(PCG64)`` for a named stream."""
    return np.random.Generator(np.random.PCG64(seed_sequence(stream, config)))


def rng_identity(config: ConfigNode | None = None) -> dict[str, object]:
    """Provenance record for the stochastic configuration."""
    cfg = config if config is not None else load_config()
    return {
        "seed": cfg.reproducibility.seed,
        "bit_generator": cfg.reproducibility.bit_generator,
        "numpy_version": numpy_version(),
    }
