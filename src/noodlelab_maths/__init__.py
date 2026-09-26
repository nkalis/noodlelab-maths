"""Maths nodes: arrays, plots, fitting and calculus. Part of the ``maths`` tier:
``pip install "noodlelab[maths]"``.

Arrays are typed by dtype family, which the editor enforces when connecting:
``NDArray[np.floating]`` accepts float16/32/64 arrays but not integer arrays,
and ``NDArray[np.number]`` accepts any numeric array.

The nodes live in submodules by topic; their ids are ``maths.<function>``
whichever module they are in.

* :mod:`.arrays`: generating and transforming arrays
* :mod:`.plot`: XY plots, histograms and heatmaps
* :mod:`.fitting`: descriptive statistics, regression, curve fitting, calibration
* :mod:`.calculus`: interpolation, integrals and derivatives
"""

from __future__ import annotations

from noodlelab.tiers import require

require("maths", "matplotlib", "numpy", "pandas", "scipy")

from .arrays import *  # noqa: E402, F403
from .calculus import *  # noqa: E402, F403
from .fitting import *  # noqa: E402, F403
from .plot import *  # noqa: E402, F403
