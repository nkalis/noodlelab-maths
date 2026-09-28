"""Fitting: descriptive statistics, polynomial and linear regression, curve
fitting and calibration (inverse prediction).

The fitting nodes return their key numbers as separate outputs (to link into
Format Values or later steps) and together as a ``summary`` dict with readable
labels, ready for the report's Add Key Values node.

Linear Regression and Curve Fit also give their parameters as uncertain values
(``slope_q``, ``intercept_q``, ``estimates``) that carry the fit's covariance,
so a result computed from several parameters of one fit has the right
uncertainty (the parameters are correlated), and an uncertainty budget names
each parameter. In a Monte Carlo evaluation they are drawn jointly from the
normal distribution with that covariance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from noodlelab import Param, Quantity, Uncertain, node, warning
from noodlelab.plugin.math import TypstMath
from noodlelab.plugin.uncertainty import correlated, real_only
from noodlelab.plugin.units import is_quantity, ureg

from .common import Confidence, Numbers, column, numeric

__all__ = [
    "curve_fit",
    "inverse_prediction",
    "linear_regression",
    "polynomial_fit",
    "statistics",
]


class Statistics(NamedTuple):
    mean: float
    std: float
    minimum: float
    maximum: float
    total: float
    count: int
    summary: dict[str, float]


@node(category="Math/Fitting")
def statistics(x: Numbers, ddof: Annotated[int, Param(min=0, max=1)] = 1) -> Statistics:
    """Summary statistics of an array, ignoring NaNs: mean, standard
    deviation, minimum, maximum, sum (``total``) and the number of values.

    `ddof` = 1 gives the sample standard deviation, 0 the population one.
    """
    values = np.asarray(real_only(x, "x"), dtype=np.float64)
    stats = (
        float(np.nanmean(values)),
        float(np.nanstd(values, ddof=ddof)),
        float(np.nanmin(values)),
        float(np.nanmax(values)),
        float(np.nansum(values)),
        int(np.count_nonzero(~np.isnan(values))),
    )
    labels = ("Mean", "Standard deviation", "Minimum", "Maximum", "Sum", "Count")
    summary = {k: v for k, v in zip(labels, stats, strict=True) if k != "Sum"}
    return Statistics(*stats, summary=summary)


class PolyFit(NamedTuple):
    coefficients: NDArray[np.float64]
    fitted: NDArray[np.float64]
    r_squared: float


@node(category="Math/Fitting", title="Polynomial Fit")
def polynomial_fit(
    x: Numbers, y: Numbers, degree: Annotated[int, Param(min=0, max=12)] = 1
) -> PolyFit:
    """Least-squares polynomial fit, highest power first. Pairs with a NaN
    are left out; ``fitted`` has a value for every x."""
    xs, ys = _finite_pairs(x, y)
    if len(xs) <= degree:
        raise ValueError(f"A degree-{degree} fit needs more than {degree} points, got {len(xs)}")
    coeffs = np.polyfit(xs, ys, degree)
    fitted = np.polyval(coeffs, np.asarray(x, dtype=np.float64)).astype(np.float64)
    ss_res = float(np.sum((ys - np.polyval(coeffs, xs)) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    return PolyFit(coeffs.astype(np.float64), fitted, 1.0 - ss_res / ss_tot if ss_tot else 1.0)


@polynomial_fit.check
def _check_polyfit(degree: int = 1):
    if degree > 6:
        return warning(f"A degree-{degree} polynomial is prone to overfitting")


def _finite_pairs(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(real_only(x, "x"), dtype=np.float64)
    y = np.asarray(real_only(y, "y"), dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y differ in length ({x.size} and {y.size})")
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


# --- linear regression -----------------------------------------------------------------------


Column = Annotated[
    Quantity | Numbers,
    Param(description="Numbers, or a quantity holding them (a column with a unit)"),
]


def _magnitudes(values: Any) -> tuple[Any, Any]:
    """The numbers of a column and its unit (None for plain numbers)."""
    if is_quantity(values):
        return np.asarray(values.magnitude, dtype=np.float64), values.units
    return values, None


class Regression(NamedTuple):
    slope: float
    intercept: float
    slope_se: float
    intercept_se: float
    r_squared: float
    p_value: float
    residual_std: float
    n: int
    fitted: NDArray[np.float64]
    residuals: NDArray[np.float64]
    summary: dict[str, Any]
    slope_q: Uncertain
    intercept_q: Uncertain


@node(category="Math/Fitting", title="Linear Regression", sample=False)
def linear_regression(
    x: Column, y: Column, through_origin: bool = False, confidence: Confidence = 0.95
) -> Regression:
    """Ordinary least squares y = slope·x + intercept, with standard errors,
    R², the p-value of the slope and the residual standard deviation. Pairs
    with a NaN are left out. ``through_origin`` fixes the intercept at zero.

    ``slope_q`` and ``intercept_q`` are the slope and intercept as uncertain
    values, correlated as the fit makes them, in the units of y/x and y when
    the columns have units: ``slope_q · x + intercept_q`` has the standard
    error of the line at x."""
    from scipy import stats as st

    x, x_unit = _magnitudes(x)
    y, y_unit = _magnitudes(y)
    xs, ys = _finite_pairs(x, y)
    n = len(xs)
    k = 1 if through_origin else 2
    if n <= k:
        raise ValueError(f"A regression needs more than {k} points, got {n}")
    if through_origin:
        slope = float(np.sum(xs * ys) / np.sum(xs**2))
        intercept = 0.0
    else:
        slope, intercept = (float(v) for v in np.polyfit(xs, ys, 1))
    fitted_all = slope * np.asarray(x, dtype=np.float64) + intercept
    res = ys - (slope * xs + intercept)
    dof = n - k
    s = float(np.sqrt(np.sum(res**2) / dof))
    if through_origin:
        slope_se = s / float(np.sqrt(np.sum(xs**2)))
        intercept_se = 0.0
        ss_tot = float(np.sum(ys**2))
    else:
        sxx = float(np.sum((xs - xs.mean()) ** 2))
        slope_se = s / np.sqrt(sxx)
        intercept_se = s * float(np.sqrt(1 / n + xs.mean() ** 2 / sxx))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    # cov(slope, intercept) = -x̄·s²/Sxx: a steeper line crosses x = 0 lower
    covariance = -float(xs.mean()) * s**2 / sxx if not through_origin else 0.0
    slope_q, intercept_q = correlated(
        [slope, intercept],
        [[slope_se**2, covariance], [covariance, intercept_se**2]],
        ["slope", "intercept"],
    )
    if x_unit is not None or y_unit is not None:
        reg = ureg()
        yu = y_unit if y_unit is not None else reg.dimensionless
        xu = x_unit if x_unit is not None else reg.dimensionless
        slope_q, intercept_q = reg.Quantity(slope_q, yu / xu), reg.Quantity(intercept_q, yu)
    r2 = 1.0 - float(np.sum(res**2)) / ss_tot if ss_tot else 1.0
    t = slope / slope_se if slope_se else np.inf
    p = float(2 * st.t.sf(abs(t), dof))
    tq = float(st.t.ppf(0.5 + confidence / 2, dof))
    pct = f"{confidence:.0%}"
    summary: dict[str, Any] = {
        "Slope": slope,
        "Slope standard error": slope_se,
        f"Slope {pct} CI": f"{slope - tq * slope_se:.4g} to {slope + tq * slope_se:.4g}",
    }
    if not through_origin:
        summary["Intercept"] = intercept
        summary["Intercept standard error"] = intercept_se
    summary |= {
        "R²": r2,
        "p-value (slope)": p,
        "Residual standard deviation": s,
        "Points": n,
    }
    return Regression(
        slope,
        intercept,
        float(slope_se),
        float(intercept_se),
        r2,
        p,
        s,
        n,
        fitted_all.astype(np.float64),
        (np.asarray(y, dtype=np.float64) - fitted_all).astype(np.float64),
        summary,
        slope_q,
        intercept_q,
    )


# --- curve fitting ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Model:
    params: tuple[str, ...]
    fn: Any
    guess: Any
    equation: str  # Typst math


def _guess_exp(x: np.ndarray, y: np.ndarray) -> list[float]:
    return [float(y[np.argmax(np.abs(y))]) or 1.0, 1.0 / (float(np.ptp(x)) or 1.0), 0.0]


def _guess_gauss(x: np.ndarray, y: np.ndarray) -> list[float]:
    i = int(np.argmax(y))
    base = float(np.min(y))
    half = x[y - base >= (y[i] - base) / 2]
    width = float(np.ptp(half)) / 2.355 if half.size > 1 else float(np.ptp(x)) / 6
    return [float(y[i] - base), float(x[i]), width or 1.0, base]


def _guess_logistic(x: np.ndarray, y: np.ndarray) -> list[float]:
    return [float(np.max(y)), 4.0 / (float(np.ptp(x)) or 1.0), float(np.median(x))]


def _guess_power(x: np.ndarray, y: np.ndarray) -> list[float]:
    ok = (x > 0) & (y > 0)
    if ok.sum() >= 2:
        b, log_a = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
        return [float(np.exp(log_a)), float(b)]
    return [1.0, 1.0]


def _guess_mm(x: np.ndarray, y: np.ndarray) -> list[float]:
    vmax = float(np.max(y))
    half = x[y >= vmax / 2]
    return [vmax * 1.2, float(np.min(half)) if half.size else float(np.median(x))]


def _guess_sine(x: np.ndarray, y: np.ndarray) -> list[float]:
    y0 = y - np.mean(y)
    dt = float(np.median(np.diff(np.sort(x)))) or 1.0
    spectrum = np.abs(np.fft.rfft(y0))
    freq = np.fft.rfftfreq(len(y0), dt)
    f = float(freq[1 + int(np.argmax(spectrum[1:]))]) if len(freq) > 1 else 1.0
    return [float(np.max(np.abs(y0))), 0.1 / (float(np.ptp(x)) or 1.0), 2 * np.pi * f, 0.0]


MODELS: dict[str, Model] = {
    "linear": Model(
        ("a", "b"), lambda x, a, b: a * x + b, lambda x, y: list(np.polyfit(x, y, 1)), "y = a x + b"
    ),
    "quadratic": Model(
        ("a", "b", "c"),
        lambda x, a, b, c: a * x**2 + b * x + c,
        lambda x, y: list(np.polyfit(x, y, 2)),
        "y = a x^2 + b x + c",
    ),
    "exponential decay": Model(
        ("A", "k", "C"),
        lambda x, A, k, C: A * np.exp(-k * x) + C,
        _guess_exp,
        "y = A e^(-k x) + C",
    ),
    "exponential growth": Model(
        ("A", "k"),
        lambda x, A, k: A * np.exp(k * x),
        lambda x, y: _guess_exp(x, y)[:2],
        "y = A e^(k x)",
    ),
    "gaussian": Model(
        ("A", "mu", "sigma", "C"),
        lambda x, A, mu, sigma, C: A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + C,
        _guess_gauss,
        "y = A exp(-(x - mu)^2 / (2 sigma^2)) + C",
    ),
    "logistic": Model(
        ("L", "k", "x0"),
        lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0))),
        _guess_logistic,
        "y = L / (1 + e^(-k (x - x_0)))",
    ),
    "power law": Model(
        ("a", "b"),
        lambda x, a, b: a * np.power(x, b),
        _guess_power,
        "y = a x^b",
    ),
    "michaelis-menten": Model(
        ("Vmax", "Km"),
        lambda x, Vmax, Km: Vmax * x / (Km + x),
        _guess_mm,
        "v = (V_max [S]) / (K_m + [S])",
    ),
    "damped sine": Model(
        ("A", "gamma", "omega", "phi"),
        lambda x, A, gamma, omega, phi: A * np.exp(-gamma * x) * np.sin(omega * x + phi),
        _guess_sine,
        "y = A e^(-gamma x) sin(omega x + phi)",
    ),
}

ModelName = Literal[
    "linear",
    "quadratic",
    "exponential decay",
    "exponential growth",
    "gaussian",
    "logistic",
    "power law",
    "michaelis-menten",
    "damped sine",
]


class CurveFit(NamedTuple):
    parameters: pd.DataFrame
    values: dict[str, float]
    fitted: NDArray[np.float64]
    residuals: NDArray[np.float64]
    r_squared: float
    rmse: float
    curve_x: NDArray[np.float64]
    curve_y: NDArray[np.float64]
    equation: str
    summary: dict[str, Any]
    estimates: dict[str, Uncertain]


@node(category="Math/Fitting", title="Curve Fit", sample=False)
def curve_fit(
    x: Numbers,
    y: Numbers,
    model: ModelName = "exponential decay",
    initial: Annotated[
        str, Param(description="Starting values, comma-separated; empty: estimated from the data")
    ] = "",
    sigma: Numbers | None = None,
    confidence: Confidence = 0.95,
    curve_points: Annotated[int, Param(min=10, max=10_000)] = 300,
) -> CurveFit:
    """Non-linear least squares fit of a standard model (scipy.optimize.curve_fit).

    Outputs the parameter table (value, standard error, confidence interval),
    the fitted values and residuals at x, R² and RMSE, a smooth curve over the
    data's range for plotting, and the model's equation in Typst math for the
    report. ``sigma``: measurement uncertainties of y, used as weights.

    ``estimates`` has each parameter as an uncertain value, by name, correlated
    as the fit's covariance says (empty when the fit leaves a parameter
    undetermined, with an infinite standard error).
    """
    from scipy import optimize
    from scipy import stats as st

    m = MODELS[model]
    xs, ys = _finite_pairs(x, y)
    if len(xs) <= len(m.params):
        raise ValueError(
            f"The {model} model has {len(m.params)} parameters and needs more points "
            f"than that, got {len(xs)}"
        )
    p0 = [float(v) for v in initial.split(",")] if initial.strip() else m.guess(xs, ys)
    if len(p0) != len(m.params):
        raise ValueError(f"{model} takes {len(m.params)} starting values ({', '.join(m.params)})")
    weights = None
    if sigma is not None:
        s = np.asarray(sigma, dtype=np.float64)
        xa, ya = np.asarray(x, float), np.asarray(y, float)
        if s.shape != xa.shape:  # indexing it by the pairs kept would fail, or misalign
            raise ValueError(f"sigma has {s.size} values, but x and y have {xa.size}")
        weights = s[np.isfinite(xa) & np.isfinite(ya)]  # the same points _finite_pairs kept
        if not (np.isfinite(weights).all() and (weights > 0).all()):
            raise ValueError("sigma must be a positive number for every point fitted")
    popt, pcov = optimize.curve_fit(
        m.fn, xs, ys, p0=p0, sigma=weights, absolute_sigma=weights is not None, maxfev=20_000
    )
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    try:
        estimates = dict(zip(m.params, correlated(popt, pcov, m.params), strict=True))
    except ValueError:  # an undetermined parameter (infinite covariance)
        estimates = {}
    dof = max(1, len(xs) - len(popt))
    tq = float(st.t.ppf(0.5 + confidence / 2, dof))
    pct = f"{confidence:.0%}"
    table = pd.DataFrame(
        {
            "parameter": m.params,
            "value": popt,
            "std_error": perr,
            f"ci{pct[:-1]}_low": popt - tq * perr,
            f"ci{pct[:-1]}_high": popt + tq * perr,
            "relative_error_%": np.abs(perr / popt) * 100,
        }
    )
    xa = np.asarray(x, dtype=np.float64)
    fitted = np.asarray(m.fn(xa, *popt), dtype=np.float64)
    res_fit = ys - m.fn(xs, *popt)
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - float(np.sum(res_fit**2)) / ss_tot if ss_tot else 1.0
    rmse = float(np.sqrt(np.mean(res_fit**2)))
    cx = np.linspace(float(xs.min()), float(xs.max()), curve_points)
    summary: dict[str, Any] = {
        "Model": model,
        **{
            f"{p} ({pct} CI)": f"{v:.4g} ± {tq * e:.2g}"
            for p, v, e in zip(m.params, popt, perr, strict=True)
        },
        "R²": r2,
        "RMSE": rmse,
        "Points": len(xs),
    }
    return CurveFit(
        table,
        {p: float(v) for p, v in zip(m.params, popt, strict=True)},
        fitted,
        (np.asarray(y, dtype=np.float64) - fitted).astype(np.float64),
        r2,
        rmse,
        cx,
        np.asarray(m.fn(cx, *popt), dtype=np.float64),
        TypstMath(m.equation),
        summary,
        estimates,
    )


@curve_fit.check
def _check_curve_fit(model: str = "exponential decay", initial: str = ""):
    if not initial.strip():
        return None
    try:
        values = [float(v) for v in initial.split(",")]
    except ValueError:
        return "Starting values must be numbers separated by commas"
    params = MODELS[model].params if model in MODELS else ()
    if params and len(values) != len(params):
        return f"{model} takes {len(params)} starting values: {', '.join(params)}"
    return None


# --- calibration -------------------------------------------------------------------------------


class Calibration(NamedTuple):
    results: pd.DataFrame
    lod: float
    loq: float
    summary: dict[str, Any]


@node(category="Math/Fitting", title="Inverse Prediction", sample=False)
def inverse_prediction(
    x: Numbers,
    y: Numbers,
    samples: pd.DataFrame,
    response: Annotated[str, column("samples", description="The measured signal column")] = "",
    replicates: Annotated[int, Param(min=1, max=100, description="Readings per sample")] = 1,
    confidence: Confidence = 0.95,
) -> Calibration:
    """Calibration: fit the standards (x = known amount, y = signal) with a
    straight line, then estimate the amount in each sample from its signal.

    For each sample: the estimate x̂ = (y₀ − b)/m, its standard error
    s = (s_y/m)·√(1/k + 1/n + (y₀ − ȳ)²/(m²·Sxx)) with k replicates, a
    confidence interval, and a flag when x̂ is outside the calibrated range or
    below the limit of quantification. LOD = 3.3·s_y/m and LOQ = 10·s_y/m,
    with s_y the residual standard deviation (ICH Q2).
    """
    from scipy import stats as st

    xs, ys = _finite_pairs(x, y)
    n = len(xs)
    if n < 3:
        raise ValueError("A calibration needs at least 3 standards")
    m, b = (float(v) for v in np.polyfit(xs, ys, 1))
    res = ys - (m * xs + b)
    sy = float(np.sqrt(np.sum(res**2) / (n - 2)))
    sxx = float(np.sum((xs - xs.mean()) ** 2))
    y0 = numeric(samples, response)
    xhat = (y0 - b) / m
    se = abs(sy / m) * np.sqrt(1 / replicates + 1 / n + (y0 - ys.mean()) ** 2 / (m**2 * sxx))
    tq = float(st.t.ppf(0.5 + confidence / 2, n - 2))
    lod, loq = 3.3 * sy / abs(m), 10 * sy / abs(m)
    lo, hi = float(xs.min()), float(xs.max())
    flag = np.where(
        xhat < lod,
        "below LOD",
        np.where(xhat < loq, "below LOQ", np.where((xhat < lo) | (xhat > hi), "outside range", "")),
    )
    results = samples.copy()
    results["estimate"] = xhat
    results["std_error"] = se
    results["ci_low"] = xhat - tq * se
    results["ci_high"] = xhat + tq * se
    results["flag"] = flag
    r2 = 1 - float(np.sum(res**2)) / float(np.sum((ys - ys.mean()) ** 2))
    summary = {
        "Slope (sensitivity)": m,
        "Intercept": b,
        "R²": r2,
        "Residual standard deviation": sy,
        "Standards": n,
        "Calibrated range": f"{lo:.4g} to {hi:.4g}",
        "Limit of detection (LOD)": lod,
        "Limit of quantification (LOQ)": loq,
        "Samples flagged": int(np.count_nonzero(flag != "")),
    }
    return Calibration(results, lod, loq, summary)


@inverse_prediction.check
def _check_inverse(response: str = ""):
    if not response.strip():
        return "Choose the response column of the samples table"
