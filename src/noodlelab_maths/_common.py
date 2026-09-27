"""Types and helpers shared by the maths and science modules."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeVar

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from noodlelab import FileRef, Param
from noodlelab.core.uncertainty import real_only

CsvFile = Annotated[FileRef, Param(accept=(".csv", ".tsv", ".txt"))]

Floats = NDArray[np.floating]
Numbers = NDArray[np.number]
Confidence = Annotated[float, Param(min=0.5, max=0.999, step=0.01, precision=3)]
Colormap = Literal["viridis", "plasma", "cividis", "magma", "coolwarm", "RdBu_r", "YlOrRd", "Blues"]

# Table operations are generic: what comes out has the type of what went in,
# so a GeoDataFrame filtered, sorted or joined is still a GeoDataFrame.
TableT = TypeVar("TableT", bound=pd.DataFrame)


def column(table_input: str = "table", **kwargs: Any) -> Param:
    """A dropdown of the columns of whatever is linked into ``table_input``."""
    return Param(options_from=f"{table_input}.columns", **kwargs)


def axes(width: float = 6.0, height: float = 3.6) -> tuple[Figure, Axes]:
    # Figure, never pyplot: nodes draw in worker threads
    fig = Figure(figsize=(width, height), layout="constrained")
    return fig, fig.add_subplot()


def names(text: str) -> list[str]:
    """``"a, b ,c"`` as ``["a", "b", "c"]``."""
    return [c.strip() for c in text.split(",") if c.strip()]


def require_columns(table: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c and c not in table.columns]
    if missing:
        have = ", ".join(map(str, table.columns))
        raise KeyError(f"No column {', '.join(repr(c) for c in missing)}; the table has: {have}")


def numeric(table: pd.DataFrame, col: str) -> NDArray[np.float64]:
    """A column as float64, with a clear error for text columns."""
    require_columns(table, col)
    try:
        values = pd.to_numeric(table[col], errors="raise")
    except (TypeError, ValueError):
        raise TypeError(f"Column '{col}' is not numeric") from None
    return real_only(values, f"Column '{col}'").to_numpy(dtype=np.float64)


def as_years(values: Any) -> NDArray[np.float64]:
    """Datetimes as decimal years (2020-07-02 ≈ 2020.5); numbers unchanged."""
    s = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(s):
        t = pd.DatetimeIndex(s)
        start = pd.to_datetime(t.year.astype(str) + "-01-01")
        days = np.where(t.is_leap_year, 366.0, 365.0)
        return (t.year + (t - start).total_seconds() / 86400.0 / days).to_numpy(np.float64)
    return real_only(pd.to_numeric(s), "The values").to_numpy(np.float64)


def finish(ax: Axes, title: str, x_label: str, y_label: str) -> None:
    ax.set(title=title, xlabel=x_label, ylabel=y_label)
    ax.grid(alpha=0.3)
