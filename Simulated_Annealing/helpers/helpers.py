from collections.abc import Callable
from numbers import Integral

import numpy as np


def as_finite_float(value: float, name: str) -> float:
    """Convert an input to a finite float."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric scalar.") from exc

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return value


def as_integer(value: int, name: str, minimum: int | None = None) -> int:
    """Validate an integer scalar."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)

    if minimum is not None and value < minimum:
        if minimum == 0:
            raise ValueError(f"{name} must be a non-negative integer.")
        if minimum == 1:
            raise ValueError(f"{name} must be a positive integer.")
        raise ValueError(f"{name} must be at least {minimum}.")

    return value


def check_annealing_inputs(
    fun: Callable[[float], float],
    x0: float,
    t0: float,
    n_steps: int,
    seed: int | None,
    count_name: str = "n_iter",
) -> tuple[float, float, int, int | None]:
    """Validate common simulated annealing inputs."""
    if not callable(fun):
        raise TypeError("fun must be callable.")

    x0 = as_finite_float(x0, "x0")
    t0 = as_finite_float(t0, "t0")

    if t0 <= 0:
        raise ValueError("t0 must be positive.")

    n_steps = as_integer(n_steps, count_name, minimum=0)

    if seed is not None:
        seed = as_integer(seed, "seed", minimum=0)

    return x0, t0, n_steps, seed


def acceptance_rule(
    x_old: float,
    value_old: float,
    x_prop: float,
    value_prop: float,
    temp: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Apply the Metropolis acceptance rule from Metropolis et al. (1953)."""
    delta_f = value_prop - value_old

    if delta_f <= 0:
        return x_prop, value_prop

    if rng.uniform() < np.exp(-delta_f / temp):
        return x_prop, value_prop

    return x_old, value_old
