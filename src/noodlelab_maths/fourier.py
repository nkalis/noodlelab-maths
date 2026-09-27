"""Fourier transforms of evenly sampled signals, with units.

A signal sampled every Δt holds frequencies up to the Nyquist frequency
1/(2Δt), resolved to 1/(N Δt) by N samples. Frequencies are in the reciprocal
of the sampling unit: Hz for seconds, 1/m for a profile along a road.

Before transforming, the mean (or a trend) is taken off and a window tapers
the ends, so a record that does not hold a whole number of periods does not
smear its peaks across the spectrum (leakage). Zero padding interpolates the
spectrum on a finer grid; it does not resolve peaks closer than 1/(N Δt).

These nodes never run on samples while editing: every other point of a
signal is a different signal, with half the Nyquist frequency.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, NamedTuple

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from noodlelab import Param, Quantity, node
from noodlelab.core.units import is_quantity, ureg

__all__ = ["fourier_transform", "inverse_fourier_transform", "power_spectrum"]

Signal = Annotated[
    Quantity | NDArray[np.number],
    Param(description="Evenly sampled values: an array, or a quantity holding one"),
]
Times = Annotated[
    Quantity | NDArray[np.floating] | None,
    Param(description="When each sample was taken, evenly spaced; else use sample_spacing"),
]
Spacing = Annotated[
    Quantity, Param(description="Time (or distance) between samples, when time is not linked")
]
Window = Literal["hann", "hamming", "blackman", "flattop", "none"]
Detrend = Literal["mean", "linear", "none"]


def _values(signal: Any) -> tuple[NDArray[Any], Any]:
    reg = ureg()
    mag, unit = (signal.magnitude, signal.units) if is_quantity(signal) else (signal, None)
    x = np.asarray(mag)
    if x.ndim != 1 or x.size < 4:
        raise ValueError(f"The signal must be a list of at least 4 values; got shape {x.shape}")
    if x.dtype.kind not in "biufc":
        raise TypeError("The signal must hold numbers")
    x = x.astype(np.complex128 if x.dtype.kind == "c" else np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("The signal has NaN or infinite values: fill or drop them first")
    return x, unit if unit is not None else reg.dimensionless


def _spacing(time: Any, sample_spacing: Any, n: int) -> Any:
    """The sample spacing, as a quantity: from the times, which must be evenly spaced."""
    reg = ureg()
    if time is None:
        if sample_spacing.magnitude <= 0:
            raise ValueError("The sample spacing must be positive")
        return sample_spacing
    t, unit = (time.magnitude, time.units) if is_quantity(time) else (time, reg.second)
    t = np.asarray(t, dtype=np.float64)
    if t.shape != (n,):
        raise ValueError(f"time has {t.size} values but the signal {n}: one time per value")
    steps = np.diff(t)
    dt = float(np.mean(steps))
    if dt <= 0 or not np.allclose(steps, dt, rtol=1e-6, atol=0):
        raise ValueError("The times are not evenly spaced: resample the signal first")
    return reg.Quantity(dt, unit)


def _per(dt: Any) -> Any:
    """The unit of frequency for a sample spacing: Hz for time, 1/unit otherwise."""
    one = 1 / ureg().Quantity(1.0, dt.units)
    return one.to("Hz") if one.check("[frequency]") else one


def _factor(dt: Any, per: Any) -> float:
    """From 1/(the spacing's unit) to the frequency unit: 1/ms to Hz is 1000."""
    return float((1 / ureg().Quantity(1.0, dt.units)).to(per.units).magnitude)


def _detrended(x: NDArray[Any], detrend: str) -> NDArray[Any]:
    if detrend == "none":
        return x
    from scipy.signal import detrend as _detrend

    return _detrend(x, type="constant" if detrend == "mean" else "linear")


def _window(name: str, n: int) -> NDArray[np.float64]:
    if name == "none":
        return np.ones(n)
    from scipy.signal import get_window

    return np.asarray(get_window(name, n, fftbins=True), dtype=np.float64)


class Fourier(NamedTuple):
    frequency: Quantity
    amplitude: Quantity
    spectrum: NDArray[np.complex128]
    phase: NDArray[np.float64]
    plot: Figure
    peak_frequencies: Quantity
    dominant_frequency: Quantity
    dominant_amplitude: Quantity
    resolution: Quantity
    summary: dict[str, float]


@node(category="Math/Fourier", title="Fourier Transform", sample=False)
def fourier_transform(
    signal: Signal,
    time: Times = None,
    sample_spacing: Spacing = Quantity(1.0, "s"),  # noqa: B008 - immutable
    window: Annotated[
        Window, Param(description="Tapers the ends against leakage; none for an exact inverse")
    ] = "hann",
    detrend: Annotated[Detrend, Param(description="Taken off before transforming")] = "mean",
    padding: Annotated[
        int, Param(min=1, max=64, description="Pad with zeros to this many times the length")
    ] = 1,
    sides: Annotated[
        Literal["one-sided", "two-sided"],
        Param(description="One-sided: frequencies from 0 (real signals); two-sided: ±"),
    ] = "one-sided",
    scaling: Annotated[
        Literal["amplitude", "rms", "raw"],
        Param(description="amplitude: a sine of amplitude a reads a; rms: a/√2; raw: |DFT|"),
    ] = "amplitude",
    peaks: Annotated[int, Param(min=0, max=50, description="How many peaks to find")] = 3,
) -> Fourier:
    """The frequencies in a signal, with the discrete Fourier transform
    (NumPy's FFT). ``amplitude`` is in the signal's unit, scaled so that a
    sine of amplitude a reads a at its frequency (with a window, exactly so
    only at a bin; flattop reads amplitudes best, hann separates peaks best).
    ``spectrum`` is the complex DFT itself, for Inverse Fourier Transform.

    ``peak_frequencies`` are the frequencies of the ``peaks`` highest local
    maxima, in order of frequency: the natural frequencies in a vibration
    record, say. Peaks within two resolutions of 0 (leakage from the mean)
    and below 0.1 % of the highest do not count. A complex signal always gives both sides."""
    x, unit = _values(signal)
    n = x.size
    dt = _spacing(time, sample_spacing, n)
    step = dt.magnitude
    w = _window(window, n)
    xw = _detrended(x, detrend) * w
    nfft = n * padding
    complex_signal = np.iscomplexobj(x)
    two = sides == "two-sided" or complex_signal
    if two:
        spectrum = np.fft.fftshift(np.fft.fft(xw, nfft))
        f = np.fft.fftshift(np.fft.fftfreq(nfft, step))
    else:
        spectrum = np.fft.rfft(xw, nfft)
        f = np.fft.rfftfreq(nfft, step)
    size = np.abs(spectrum)
    if scaling != "raw":
        size = size / np.sum(w)
        if not two:  # the negative frequencies' share, except at 0 and at Nyquist
            doubled = f > 0
            if nfft % 2 == 0:
                doubled &= f < f.max()
            size = np.where(doubled, 2 * size, size)
        if scaling == "rms":
            size = np.where(f != 0, size / math.sqrt(2), size)
    per = _per(dt)
    reg = ureg()
    frequency = reg.Quantity(f * _factor(dt, per), per.units)  # f is in 1/(dt's unit)
    amplitude = reg.Quantity(size, unit)

    # peaks: local maxima away from 0, the highest first, then listed by frequency
    from scipy.signal import find_peaks

    # a peak within two resolutions of 0 is the mean's leakage, not a frequency in the signal
    away = np.abs(f) > 2 / (n * step)
    floor = 1e-3 * float(np.max(size[away])) if np.any(away) else 0.0  # not round-off bumps
    found, _ = find_peaks(np.where(away, size, 0.0), height=floor)
    chosen = sorted(found[np.argsort(size[found])[::-1][:peaks]], key=lambda i: f[i])
    top = int(np.argmax(np.where(away, size, -np.inf))) if np.any(away) else 0
    resolution = (1 / (n * dt)).to(per.units)

    fig = Figure(figsize=(6.0, 3.4), layout="constrained")
    ax = fig.add_subplot()
    ax.plot(frequency.magnitude, size, lw=1.5)
    if chosen:
        ax.plot(frequency.magnitude[chosen], size[chosen], "v", color="#c44e52", ms=7)
        for i in chosen:
            ax.annotate(
                f"{frequency.magnitude[i]:.4g}",
                (frequency.magnitude[i], size[i]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize="small",
            )
    unit_text = f" ({unit:~P})" if f"{unit:~P}" else ""
    ax.set(
        title="Amplitude spectrum" if scaling != "raw" else "|DFT|",
        xlabel=f"frequency ({per.units:~P})",
        ylabel=f"{scaling}{unit_text}",
    )
    ax.grid(alpha=0.3)

    summary = {
        f"Peak {j + 1} ({per.units:~P})": float(f_)
        for j, f_ in enumerate(frequency.magnitude[chosen])
    }
    summary[f"Resolution ({per.units:~P})"] = float(resolution.magnitude)
    return Fourier(
        frequency,
        amplitude,
        spectrum.astype(np.complex128),
        np.angle(spectrum).astype(np.float64),
        fig,
        frequency[chosen] if chosen else reg.Quantity(np.array([]), per.units),
        frequency[top],
        amplitude[top],
        resolution,
        summary,
    )


@fourier_transform.check
def _check_fourier(sample_spacing: Any = None, time: Any = None) -> str | None:
    if time is None and sample_spacing is not None and sample_spacing.magnitude <= 0:
        return "The sample spacing must be positive"
    return None


class Inverse(NamedTuple):
    time: Quantity
    signal: NDArray[np.float64]
    imaginary: NDArray[np.float64]


@node(category="Math/Fourier", title="Inverse Fourier Transform", sample=False)
def inverse_fourier_transform(
    spectrum: NDArray[np.complexfloating],
    frequency: Quantity,
    samples: Annotated[
        int, Param(min=0, description="Length of the signal; 0: from the spectrum")
    ] = 0,
) -> Inverse:
    """The signal back from its complex spectrum (the ``spectrum`` of Fourier
    Transform, perhaps filtered on the way). One-sided or two-sided is told
    from the frequencies. The round trip is exact with window none and
    detrend none; otherwise the result is the windowed, detrended signal.
    With a one-sided spectrum of an odd-length signal, give ``samples``."""
    reg = ureg()
    x = np.asarray(spectrum, dtype=np.complex128)
    f = np.asarray(frequency.magnitude, dtype=np.float64)
    if x.shape != f.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("spectrum and frequency must be two lists of the same length")
    df = float(np.median(np.diff(np.sort(f))))
    if np.any(f < 0):
        n = samples or x.size
        signal = np.fft.ifft(np.fft.ifftshift(x), n)
    else:
        n = samples or 2 * (x.size - 1)
        signal = np.fft.irfft(x, n).astype(np.complex128)
    step = 1 / (reg.Quantity(df, frequency.units) * n)  # Δt = 1/(N Δf)
    step = step.to("s") if step.check("[time]") else step.to_reduced_units()
    time = reg.Quantity(np.arange(n) * step.magnitude, step.units)
    return Inverse(time, signal.real.astype(np.float64), signal.imag.astype(np.float64))


class PowerSpectrum(NamedTuple):
    frequency: Quantity
    psd: Quantity
    plot: Figure
    rms: Quantity
    dominant_frequency: Quantity
    summary: dict[str, float]


@node(category="Math/Fourier", title="Power Spectrum", sample=False)
def power_spectrum(
    signal: Signal,
    time: Times = None,
    sample_spacing: Spacing = Quantity(1.0, "s"),  # noqa: B008 - immutable
    method: Annotated[
        Literal["welch", "periodogram"],
        Param(description="welch: averaged over overlapping segments, less noisy"),
    ] = "welch",
    window: Window = "hann",
    segment: Annotated[
        int, Param(min=8, description="Welch: samples per segment (finer frequency: longer)")
    ] = 1024,
    scaling: Annotated[
        Literal["density", "spectrum"],
        Param(description="density: power per Hz (noise); spectrum: power per peak (tones)"),
    ] = "density",
) -> PowerSpectrum:
    """How the power of a signal is spread over frequency: its power spectral
    density (unit² per Hz), or with ``spectrum`` scaling the power of each
    tone (unit²). ``rms`` is the signal's root mean square about its mean,
    from the whole spectrum (Parseval)."""
    from scipy.signal import periodogram, welch

    x, unit = _values(signal)
    if np.iscomplexobj(x):
        raise TypeError("The power spectrum here is for real signals")
    dt = _spacing(time, sample_spacing, x.size)
    fs = 1 / dt.magnitude
    win = "boxcar" if window == "none" else window
    if method == "welch":
        f, p = welch(x, fs, window=win, nperseg=min(segment, x.size), scaling=scaling)
        _, dens = welch(x, fs, window=win, nperseg=min(segment, x.size), scaling="density")
    else:
        f, p = periodogram(x, fs, window=win, scaling=scaling)
        _, dens = periodogram(x, fs, window=win, scaling="density")
    per = _per(dt)
    reg = ureg()
    to_per = _factor(dt, per)
    frequency = reg.Quantity(f * to_per, per.units)
    df = float(f[1] - f[0]) if f.size > 1 else 0.0
    rms = reg.Quantity(math.sqrt(float(np.sum(dens)) * df), unit)
    psd = (
        reg.Quantity(p / to_per, unit**2 / per.units)
        if scaling == "density"
        else (reg.Quantity(p, unit**2))
    )
    top = int(np.argmax(np.where(f > 0, p, -np.inf))) if f.size > 1 else 0
    fig = Figure(figsize=(6.0, 3.4), layout="constrained")
    ax = fig.add_subplot()
    ax.semilogy(frequency.magnitude[1:], np.maximum(psd.magnitude[1:], 1e-300), lw=1.5)
    ax.set(
        title="Power spectral density" if scaling == "density" else "Power spectrum",
        xlabel=f"frequency ({per.units:~P})",
        ylabel=f"{psd.units:~P}" if f"{psd.units:~P}" else "power",
    )
    ax.grid(alpha=0.3, which="both")
    return PowerSpectrum(
        frequency,
        psd,
        fig,
        rms,
        frequency[top],
        {"RMS": float(rms.magnitude), f"Dominant ({per.units:~P})": float(frequency[top].m)},
    )
