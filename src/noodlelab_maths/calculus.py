"""Calculus on sampled data: interpolation, integrals and derivatives."""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from noodlelab import node

from .common import Floats, Numbers

__all__ = [
    "derivative",
    "integrate",
    "interpolate",
]


@node(category="Math/Calculus")
def interpolate(
    x: Numbers,
    y: Numbers,
    new_x: Numbers,
    method: Literal["linear", "cubic", "nearest"] = "linear",
) -> Floats:
    """Values of y at new positions: resample onto another grid, or fill gaps.
    Positions outside the data give NaN."""
    from scipy.interpolate import interp1d

    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(xv) & np.isfinite(yv)
    order = np.argsort(xv[ok])
    f = interp1d(xv[ok][order], yv[ok][order], kind=method, bounds_error=False)
    return np.asarray(f(np.asarray(new_x, dtype=np.float64)), dtype=np.float64)


class Integral(NamedTuple):
    total: float
    cumulative: NDArray[np.float64]


@node(category="Math/Calculus")
def integrate(x: Numbers, y: Numbers) -> Integral:
    """The area under y(x) by the trapezoidal rule, and its running total
    (e.g. rainfall rate to accumulated rain, velocity to distance)."""
    from scipy.integrate import cumulative_trapezoid

    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    cumulative = cumulative_trapezoid(yv, xv, initial=0.0)
    return Integral(float(cumulative[-1]) if len(cumulative) else 0.0, cumulative)


@node(category="Math/Calculus")
def derivative(x: Numbers, y: Numbers) -> Floats:
    """dy/dx by central differences (numpy.gradient), for uneven spacing too."""
    return np.gradient(np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64))
