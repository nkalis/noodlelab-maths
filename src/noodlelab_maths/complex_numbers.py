"""Complex numbers: their parts, with units.

An impedance, a phasor or a Fourier spectrum is complex, and most nodes work
on real numbers only: they refuse a complex value rather than keep its real
part without a word. Complex Parts splits one into the real numbers those
nodes take: its real and imaginary parts, its size (modulus) and its phase.
The parts keep the unit (the phase is in radians), so the magnitude of an
impedance in ohms is in ohms too.
"""

from __future__ import annotations

from typing import Annotated, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from noodlelab import Param, Quantity, node
from noodlelab.plugin.units import is_quantity, ureg

__all__ = ["complex_parts"]

Radians = Quantity["rad"]


class ComplexParts(NamedTuple):
    real: Any
    imag: Any
    magnitude: Any
    phase: Radians
    conjugate: Any


@node(category="Math/Complex", title="Complex Parts", fold=True, vectorized=True)
def complex_parts(
    value: Annotated[
        Quantity | NDArray[np.number] | complex,
        Param(description="A complex number or array, with a unit or without: 3+4j ohm"),
    ] = Quantity(3 + 4j, "ohm"),  # noqa: B008 - immutable
) -> ComplexParts:
    """The parts of a complex number or array: ``real`` and ``imag``,
    ``magnitude`` |z| and ``phase`` arg z (in radians, from −π to π), and
    the ``conjugate``. Each keeps the value's unit, except the phase. A real
    value gives itself, an imaginary part of 0 and a phase of 0 or π."""
    reg = ureg()
    mag, unit = (value.magnitude, value.units) if is_quantity(value) else (value, None)
    z = np.asarray(mag)
    if z.dtype.kind not in "biufc":
        raise TypeError(f"Complex Parts takes numbers, not {z.dtype} values")

    def keep(part: Any) -> Any:
        part = part.item() if np.ndim(part) == 0 else part
        return reg.Quantity(part, unit) if unit is not None else part

    phase = np.angle(z)
    return ComplexParts(
        real=keep(np.real(z).astype(np.float64)),
        imag=keep(np.imag(z).astype(np.float64)),
        magnitude=keep(np.abs(z).astype(np.float64)),
        phase=reg.Quantity(phase.item() if np.ndim(phase) == 0 else phase, "rad"),
        conjugate=keep(np.conjugate(z)),
    )
