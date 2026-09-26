"""Generating and transforming arrays."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray

from noodlelab import Param, node, warning

from ._common import Floats, Numbers

__all__ = [
    "add_noise",
    "apply_function",
    "array_math",
    "clip_values",
    "cumulative_sum",
    "linspace",
]


@node(category="Math/Arrays")
def linspace(
    start: float = 0.0,
    stop: float = 10.0,
    num: Annotated[int, Param(min=2, max=1_000_000)] = 200,
) -> NDArray[np.float64]:
    """Evenly spaced numbers over an interval."""
    return np.linspace(start, stop, num)


@linspace.check
def _check_linspace(start: float = 0.0, stop: float = 10.0):
    if start == stop:
        return warning("Start and stop are equal, so every value is the same")


@node(category="Math/Arrays", title="Array Math", vectorized=True)
def array_math(
    a: Floats,
    b: float | Floats = 1.0,
    operation: Literal["add", "subtract", "multiply", "divide", "power"] = "multiply",
) -> Floats:
    """Element-wise math between an array and a scalar or another array."""
    ops = {
        "add": np.add,
        "subtract": np.subtract,
        "multiply": np.multiply,
        "divide": np.divide,
        "power": np.power,
    }
    return np.asarray(ops[operation](a, b), dtype=np.result_type(a, np.float64))


@array_math.check
def _check_array_math(operation: str = "multiply", b: float | None = None):
    if operation == "divide" and b == 0:
        return "Dividing by zero"


@node(category="Math/Arrays", title="Apply Function", vectorized=True)
def apply_function(
    x: Floats,
    function: Literal["sin", "cos", "exp", "log", "log10", "sqrt", "abs", "gaussian"] = "sin",
) -> Floats:
    """Apply a common function element-wise."""
    if function == "gaussian":
        return np.exp(-0.5 * x**2)
    return getattr(np, function)(x)


@node(category="Math/Arrays", title="Add Noise")
def add_noise(
    x: Floats,
    sigma: Annotated[float, Param(min=0.0, step=0.01, precision=3)] = 0.1,
    seed: int = 0,
) -> Floats:
    """Add Gaussian noise. A fixed seed keeps runs reproducible (and cacheable)."""
    rng = np.random.default_rng(seed)
    return x + rng.normal(0.0, sigma, size=np.shape(x))


@node(category="Math/Arrays", title="Clip Values")
def clip_values(x: Numbers, minimum: float = 0.0, maximum: float = 1.0) -> Floats:
    """Limit values to [minimum, maximum]."""
    return np.clip(np.asarray(x, dtype=np.float64), minimum, maximum)


@clip_values.check
def _check_clip(minimum: float = 0.0, maximum: float = 1.0):
    if minimum > maximum:
        return "The minimum is larger than the maximum"


@node(category="Math/Arrays", title="Cumulative Sum")
def cumulative_sum(x: Numbers) -> Floats:
    """Running total of an array (NaNs count as zero)."""
    return np.nancumsum(np.asarray(x, dtype=np.float64))
