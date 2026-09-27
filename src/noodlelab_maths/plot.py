"""Plots: line plots, histograms and heatmaps as Matplotlib figures.

Figures are built with the object-oriented API (``Figure``, never pyplot), so
nodes can draw in worker threads. They show as images under the node and are
embedded as vector graphics by the report's Add Figure node.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from noodlelab import Param, RunContext, node
from noodlelab.core import figures, uncertainty
from noodlelab.core.uncertainty import real_only

from ._common import Colormap, Numbers, UncertainArray, axes, finish

__all__ = [
    "heatmap",
    "histogram_plot",
    "save_figure",
    "xy_plot",
]


@node(category="Math/Plot", title="XY Plot")
def xy_plot(
    x: Numbers,
    y: UncertainArray,
    y2: Numbers | None = None,
    x2: Annotated[Numbers | None, Param(description="x for y2, if it differs from x")] = None,
    error: Annotated[Numbers | None, Param(description="Error bars for y")] = None,
    style: Literal["line", "scatter", "scatter + line"] = "line",
    title: str = "",
    x_label: str = "x",
    y_label: str = "y",
    label: str = "data",
    label2: str = "fit",
    log_x: bool = False,
    log_y: bool = False,
    style2: Annotated[
        Literal["line", "markers"], Param(description="markers: e.g. to mark peaks on y")
    ] = "line",
) -> Figure:
    """Plot y (and optionally y2, e.g. a fitted curve or marked peaks) against x.
    A y with uncertainties gets error bars of ±u unless ``error`` is linked."""
    # Matplotlib would draw only the real part of a complex array
    for name, v in (("x", x), ("y", y), ("y2", y2), ("x2", x2), ("error", error)):
        real_only(v, name)
    y, u = uncertainty.split(y)
    if error is None:
        error = u
    fig, ax = axes()
    if error is not None:
        ax.errorbar(x, y, yerr=error, fmt="none", ecolor="0.5", elinewidth=1, capsize=2)
    if style in ("scatter", "scatter + line"):
        ax.scatter(x, y, s=12, alpha=0.75, label=label)
    if style in ("line", "scatter + line"):
        ax.plot(x, y, lw=1.5, label=None if style != "line" else label)
    if y2 is not None:
        if style2 == "markers":
            ax.plot(x if x2 is None else x2, y2, "v", ms=7, color="#c44e52", label=label2)
        else:
            ax.plot(x if x2 is None else x2, y2, lw=2, label=label2)
    finish(ax, title, x_label, y_label)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if y2 is not None:
        ax.legend()
    return fig


@node(category="Math/Plot", title="Histogram Plot")
def histogram_plot(
    x: Numbers,
    bins: Annotated[int, Param(min=2, max=500)] = 30,
    normal_curve: Annotated[bool, Param(description="Overlay a normal distribution")] = False,
    title: str = "",
    x_label: str = "value",
    log_y: bool = False,
) -> Figure:
    """Distribution of values, optionally with the normal curve of the same
    mean and standard deviation for comparison."""
    v = np.asarray(real_only(x, "x"), dtype=np.float64)
    v = v[np.isfinite(v)]
    fig, ax = axes()
    ax.hist(v, bins=bins, color="#4c72b0", alpha=0.8, edgecolor="white", linewidth=0.5)
    if normal_curve and len(v) > 1:
        mu, sd = float(np.mean(v)), float(np.std(v, ddof=1))
        grid = np.linspace(v.min(), v.max(), 200)
        width = (v.max() - v.min()) / bins
        pdf = np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        ax.plot(
            grid,
            pdf * len(v) * width,
            color="#c44e52",
            lw=2,
            label=f"normal (μ={mu:.3g}, σ={sd:.3g})",
        )
        ax.legend()
    if log_y:
        ax.set_yscale("log")
    finish(ax, title, x_label, "count")
    return fig


@node(category="Math/Plot")
def heatmap(
    table: pd.DataFrame,
    colormap: Colormap = "RdBu_r",
    annotate: bool = True,
    symmetric: Annotated[
        bool, Param(description="Centre the colours on zero (for correlations, anomalies)")
    ] = True,
    title: str = "",
    colorbar_label: str = "",
) -> Figure:
    """A matrix as coloured cells, such as a correlation matrix. A first text
    column names the rows; the other columns must be numeric."""
    data = table
    row_labels = [str(i) for i in table.index]
    if len(table.columns) and not pd.api.types.is_numeric_dtype(table[table.columns[0]]):
        row_labels = table[table.columns[0]].astype(str).tolist()
        data = table.drop(columns=table.columns[0])
    values = data.to_numpy(dtype=np.float64)
    n_rows, n_cols = values.shape
    fig, ax = axes(max(4.0, 0.6 * n_cols + 2), max(3.0, 0.45 * n_rows + 1.2))
    kwargs = {}
    if symmetric:
        m = float(np.nanmax(np.abs(values))) or 1.0
        kwargs = {"vmin": -m, "vmax": m}
    im = ax.imshow(values, cmap=colormap, aspect="auto", **kwargs)
    ax.set_xticks(range(n_cols), [str(c) for c in data.columns], rotation=45, ha="right")
    ax.set_yticks(range(n_rows), row_labels)
    if annotate and n_rows * n_cols <= 400:
        for i in range(n_rows):
            for j in range(n_cols):
                v = values[i, j]
                if np.isfinite(v):
                    dark = abs(v) > 0.6 * (kwargs.get("vmax") or np.nanmax(np.abs(values)))
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if dark else "black",
                    )
    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_title(title)
    return fig


@node(category="Output", title="Save Figure")
def save_figure(
    figure: Figure,
    ctx: RunContext,
    filename: str = "figure.png",
    dpi: Annotated[int, Param(min=50, max=600)] = 200,
) -> Path:
    """Write a figure into this run's output folder. The extension picks the
    format: .png, .svg or .pdf."""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".png", ".svg", ".pdf", ".jpg"):
        filename += ".png"
    out = ctx.path(filename)
    figures.save(figure, out, dpi=dpi)
    ctx.log(f"Wrote {out.name}")
    return out
