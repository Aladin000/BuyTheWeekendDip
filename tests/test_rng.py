"""Deterministic random number generation.

The project spec locks the seed ``20260901``, NumPy ``Generator(PCG64)``, and
recording the NumPy version.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import ConfigError
from src.rng import make_rng, numpy_version, rng_identity, seed_sequence, stream_key


def test_s1_18_generator_uses_pcg64(config) -> None:
    rng = make_rng("unit-test", config)
    assert isinstance(rng, np.random.Generator)
    assert isinstance(rng.bit_generator, np.random.PCG64)
    assert type(rng.bit_generator).__name__ == config.reproducibility.bit_generator


def test_s1_19_same_stream_reproduces_identical_draws(config) -> None:
    first = make_rng("month-block-bootstrap", config).integers(0, 1_000_000, size=32)
    second = make_rng("month-block-bootstrap", config).integers(0, 1_000_000, size=32)
    assert np.array_equal(first, second)


def test_s1_19_different_streams_diverge(config) -> None:
    bootstrap = make_rng("month-block-bootstrap", config).integers(0, 1_000_000, size=32)
    other = make_rng("leave-one-out", config).integers(0, 1_000_000, size=32)
    assert not np.array_equal(bootstrap, other)


def test_s1_19_stream_key_is_stable_across_processes(config) -> None:
    """A fixed SHA-256 derivation, not Python's randomised hash."""
    assert stream_key("month-block-bootstrap") == stream_key("month-block-bootstrap")
    assert stream_key("a") != stream_key("b")
    with pytest.raises(ConfigError, match="non-empty string"):
        stream_key("")


def test_s1_20_seed_comes_from_configuration(config) -> None:
    assert config.reproducibility.seed == 20260901
    sequence = seed_sequence("unit-test", config)
    assert sequence.entropy == config.reproducibility.seed
    assert sequence.spawn_key == (stream_key("unit-test"),)


def test_s1_20_default_tolerances_are_locked(config) -> None:
    assert config.reproducibility.atol == config.reproducibility.rtol == 1e-12


def test_s1_21_numpy_version_is_recorded(config) -> None:
    assert config.reproducibility.record_numpy_version is True
    assert numpy_version() == np.__version__

    identity = rng_identity(config)
    assert identity == {
        "seed": 20260901,
        "bit_generator": "PCG64",
        "numpy_version": np.__version__,
    }
