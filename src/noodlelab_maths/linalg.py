"""Matrices: building them, solving A x = b, and eigenvalues, with units.

A matrix is a quantity holding a 2-D array, with one unit for every entry: a
stiffness matrix in N/m, a mass matrix in kg. Plain arrays are matrices without
a unit. Units follow the algebra, so K⁻¹ F of a stiffness and a force is a
displacement, and the eigenvalues of K against M are in 1/s².

A matrix whose entries have different units (a state-space A holding both
1/s and 1/s², say) is not supported: Pint gives a quantity one unit. Work in
consistent SI units and plain numbers for those.

Rows and columns are numbered from 1, as in x1, x2 and mode 1. A vector
(a 1-D array) counts as a column wherever a matrix is expected.

Values with uncertainties are not propagated through a matrix: put the
calculation in a Monte Carlo zone instead, which runs these nodes once per trial.
Nor do they run on samples while editing: every other row of a matrix is not
a smaller matrix with the same answer.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Any, Literal, NamedTuple

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from noodlelab import Param, Quantity, node, warning
from noodlelab.core import uncertainty
from noodlelab.core.units import dims_or_none, is_quantity, ureg

__all__ = [
    "determinant",
    "diagonal_matrix",
    "eigenvalues",
    "element",
    "identity_matrix",
    "inverse",
    "matrix",
    "matrix_multiply",
    "natural_frequencies",
    "solve_linear",
    "transpose",
]

MATRIX_HELP = (
    "A matrix: a quantity holding a 2-D array (one unit for every entry), or plain numbers"
)
Matrix = Annotated[Quantity | NDArray[np.floating], Param(description=MATRIX_HELP)]
OptionalMatrix = Annotated[Quantity | NDArray[np.floating] | None, Param(description=MATRIX_HELP)]
Unit = Annotated[str, Param(description="One unit for every entry; empty: plain numbers")]
Hertz = Quantity["Hz"]
RadPerSecond = Quantity["rad/s"]


# --- helpers ----------------------------------------------------------------------------------


def _split(
    value: Any,
    name: str,
    *,
    square: bool = False,
    vector_ok: bool = True,
    complex_ok: bool = False,
) -> Any:
    """The numbers of a matrix input as float64 (complex128 with ``complex_ok``,
    when they are complex), and its unit, with errors that say what is wrong
    with the shape. Most of these nodes take real matrices only: casting would
    keep only the real parts."""
    reg = ureg()
    if is_quantity(value):
        mag, unit = value.magnitude, value.units
    else:
        mag, unit = value, reg.dimensionless
    arr = np.asarray(mag)
    if arr.dtype == object or uncertainty.is_uncertain(value):
        raise TypeError(
            f"{name} holds values with uncertainties: a matrix carries no uncertainty. "
            "Use a Monte Carlo zone to propagate them"
        )
    if arr.dtype.kind == "c" and not complex_ok:
        raise TypeError(f"{name} is complex: this node works on real matrices")
    arr = arr.astype(np.complex128 if arr.dtype.kind == "c" else np.float64)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be a matrix; got a single value ({arr.item():g})")
    if arr.ndim > 2:
        raise ValueError(f"{name} must be a matrix; got an array of {arr.ndim} dimensions")
    if arr.ndim == 1 and not vector_ok:
        raise ValueError(f"{name} must be a matrix; got a vector of {arr.size}")
    if square and (arr.ndim != 2 or arr.shape[0] != arr.shape[1]):
        raise ValueError(f"{name} must be a square matrix; it is {_shape(arr)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} has NaN or infinite entries")
    return arr, unit


def _name(unit: Any) -> str:
    return f"{unit:~P}" or "no unit"


def _shape(arr: NDArray[Any]) -> str:
    return f"{arr.shape[0]}×{arr.shape[1]}" if arr.ndim == 2 else f"a vector of {arr.size}"


def _join(mag: Any, unit: Any) -> Any:
    return ureg().Quantity(mag, unit)


_ROWS = re.compile(r"[;\n]")
_ENTRIES = re.compile(r"[,\s]+")
# a complex entry as Python writes it, without spaces: 3+4j, -2.5e-3j, (1-2j)
_COMPLEX_ENTRY = re.compile(r"\(?[-+]?[\d.eE+-]*j\)?")


def _entry(token: str) -> float | complex:
    if _COMPLEX_ENTRY.fullmatch(token):
        return complex(token)
    return float(token)


def _parse_matrix(text: str) -> NDArray[np.float64] | NDArray[np.complex128]:
    """``"2, -1; -1, 2"`` (MATLAB style) as a 2-D array: rows end at ``;`` or a
    new line, entries are separated by commas or spaces. Entries may be
    complex (``3+4j``, no spaces inside); the matrix is then complex."""
    body = text.strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    rows: list[list[float | complex]] = []
    for line in _ROWS.split(body):
        tokens = [t for t in _ENTRIES.split(line.strip().strip("[]")) if t]
        if not tokens:
            continue
        row = []
        for token in tokens:
            try:
                row.append(_entry(token))
            except ValueError:
                hint = (
                    ": give the unit once, in unit, for the whole matrix"
                    if dims_or_none(token) is not None
                    else ""
                )
                where = f"row {len(rows) + 1}"
                raise ValueError(f"'{token}' in {where} is not a number{hint}") from None
        if rows and len(row) != len(rows[0]):
            raise ValueError(
                f"Row {len(rows) + 1} has {len(row)} entries, but row 1 has {len(rows[0])}"
            )
        rows.append(row)
    if not rows:
        raise ValueError("Type the rows of the matrix, such as 2, -1; -1, 2")
    complex_ = any(isinstance(v, complex) for row in rows for v in row)
    return np.array(rows, dtype=np.complex128 if complex_ else np.float64)


def _unit_problem(unit: str) -> str | None:
    if unit.strip() and dims_or_none(unit) is None:
        return f"Unknown unit '{unit}'"
    return None


def _in(unit: str) -> Any:
    return ureg().Unit(unit.strip()) if unit.strip() else ureg().dimensionless


# --- building matrices ------------------------------------------------------------------------


@node(category="Math/Matrices", title="Matrix", fold=True)
def matrix(
    text: Annotated[
        str,
        Param(
            multiline=True,
            description="Rows end at ; or a new line, entries are separated by commas or "
            "spaces: 2, -1; -1, 2",
        ),
    ] = "2, -1; -1, 2",
    unit: Unit = "",
) -> Quantity:
    """A matrix typed as text, MATLAB style: ``2, -1; -1, 2`` is a 2×2 matrix,
    ``10; 0`` a column. Every entry has the one ``unit``."""
    return _join(_parse_matrix(text), _in(unit))


@matrix.check
def _check_matrix(text: str = "", unit: str = "") -> str | None:
    try:
        _parse_matrix(text)
    except ValueError as exc:
        return str(exc)
    return _unit_problem(unit)


@node(category="Math/Matrices", title="Identity Matrix", fold=True)
def identity_matrix(size: Annotated[int, Param(min=1, max=1000)] = 2, unit: Unit = "") -> Quantity:
    """The identity matrix: ones on the diagonal, zeros elsewhere."""
    return _join(np.eye(size), _in(unit))


@identity_matrix.check
def _check_identity(unit: str = "") -> str | None:
    return _unit_problem(unit)


@node(category="Math/Matrices", title="Diagonal Matrix", sample=False, fold=True)
def diagonal_matrix(
    diagonal: Annotated[str, Param(description="The diagonal, comma separated: 1, 1")] = "1, 1",
    unit: Unit = "kg",
    values: Annotated[
        Quantity | NDArray[np.floating] | None,
        Param(description="A vector for the diagonal, instead of the text (and its unit)"),
    ] = None,
) -> Quantity:
    """A matrix with ``diagonal`` on its diagonal and zeros elsewhere, such as
    the mass matrix of masses on springs. A linked vector replaces the text."""
    if values is not None:
        arr, u = _split(values, "values")
        return _join(np.diag(arr.ravel()), u)
    return _join(np.diag(_parse_matrix(diagonal).ravel()), _in(unit))


@diagonal_matrix.check
def _check_diagonal(diagonal: str = "", unit: str = "", values: Any = None) -> str | None:
    if values is None:
        try:
            _parse_matrix(diagonal)
        except ValueError as exc:
            return str(exc)
    return _unit_problem(unit)


# --- operations -------------------------------------------------------------------------------


@node(category="Math/Matrices", title="Transpose", sample=False)
def transpose(matrix: Matrix) -> Quantity:
    """Rows become columns. A vector (a column) becomes a row, 1×n."""
    arr, unit = _split(matrix, "matrix")
    return _join(arr.reshape(-1, 1).T if arr.ndim == 1 else arr.T, unit)


@node(category="Math/Matrices", title="Matrix Multiply", sample=False)
def matrix_multiply(a: Matrix, b: Matrix) -> Quantity:
    """The matrix product A B, with the units multiplied too: a stiffness
    matrix times a displacement vector is a force vector. A vector counts as a
    column. For element-by-element products use Quantity Math."""
    x, ux = _split(a, "a")
    y, uy = _split(b, "b")
    if x.ndim == 1:
        x = x.reshape(1, -1)
    inner = y.shape[0]
    if x.shape[1] != inner:
        raise ValueError(
            f"A is {_shape(x)} and B is {_shape(y)}: the columns of A must match the rows of B"
        )
    return _join(x @ y, ux * uy).to_reduced_units()  # cm times N/m is in N


def _rank(arr: NDArray[np.float64]) -> int:
    return int(np.linalg.matrix_rank(arr))


@node(category="Math/Matrices", title="Inverse", sample=False)
def inverse(matrix: Matrix) -> Quantity:
    """The inverse A⁻¹, in the reciprocal unit: the inverse of a stiffness
    matrix (N/m) is a flexibility matrix (m/N). To solve A x = b, Solve Linear
    System is more accurate than multiplying by the inverse."""
    arr, unit = _split(matrix, "matrix", square=True, vector_ok=False)
    rank = _rank(arr)
    if rank < arr.shape[0]:
        raise ValueError(f"The matrix is singular (rank {rank} of {arr.shape[0]}): no inverse")
    return _join(np.linalg.inv(arr), 1 / unit)


@inverse.check
def _check_inverse(matrix: Any = None) -> Any:
    if matrix is None:
        return None
    try:
        arr, _ = _split(matrix, "matrix", square=True, vector_ok=False)
    except (TypeError, ValueError) as exc:
        return str(exc)
    cond = np.linalg.cond(arr)
    if math.isfinite(cond) and cond > 1e12:
        return warning(f"The matrix is nearly singular (condition number {cond:.3g})")
    return None


@node(category="Math/Matrices", title="Determinant", sample=False)
def determinant(matrix: Matrix) -> Quantity:
    """The determinant, in the matrix's unit to the power of its size. Zero
    means the matrix is singular: its equations are not independent."""
    arr, unit = _split(matrix, "matrix", square=True, vector_ok=False)
    return _join(float(np.linalg.det(arr)), unit ** arr.shape[0])


@node(category="Math/Matrices", title="Element", sample=False)
def element(
    matrix: Matrix,
    row: Annotated[int, Param(min=1, description="Numbered from 1")] = 1,
    column: Annotated[int, Param(min=1, description="Numbered from 1; 1 for a vector")] = 1,
) -> Quantity:
    """One entry of a matrix or vector, with its unit. Rows and columns are
    numbered from 1, so row 2 of a solution vector is x2. An entry of a
    complex matrix is complex."""
    arr, unit = _split(matrix, "matrix", complex_ok=True)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    rows, cols = arr.shape
    if not (1 <= row <= rows and 1 <= column <= cols):
        raise IndexError(f"No entry ({row}, {column}): the matrix is {rows}×{cols}")
    return _join(arr[row - 1, column - 1].item(), unit)


# --- solving ----------------------------------------------------------------------------------


class LinearSolution(NamedTuple):
    x: Quantity
    residual: Quantity
    condition: float
    rank: int
    method: str
    summary: dict[str, float]


@node(category="Math/Matrices", title="Solve Linear System", sample=False)
def solve_linear(
    a: Matrix,
    b: Matrix,
    unit: Annotated[str, Param(description="Unit of x; empty: b's unit over A's")] = "",
) -> LinearSolution:
    """Solve A x = b for x, such as K x = F for the displacements of a
    structure. x is in b's unit over A's (N over N/m is m), and has b's shape.

    A square A with independent rows gives the exact solution. More equations
    than unknowns give the least-squares solution, fewer the smallest one that
    fits (``method`` says which). ``residual`` is b − A x, and ``condition``
    how much errors in A and b can be amplified: above about 1e12 the answer
    means little."""
    x_a, ua = _split(a, "A")
    x_b, ub = _split(b, "b")
    if x_a.ndim == 1:
        x_a = x_a.reshape(-1, 1)
    rows, cols = x_a.shape
    if x_b.shape[0] != rows:
        raise ValueError(f"A has {rows} rows but b has {x_b.shape[0]}: one entry of b per row")
    rank = _rank(x_a)
    cond = float(np.linalg.cond(x_a))
    if rows == cols:
        if rank < rows:
            raise ValueError(
                f"The equations are not independent (rank {rank} of {rows}): "
                "there is no single solution"
            )
        x = np.linalg.solve(x_a, x_b)
        method = "exact"
    else:
        x = np.linalg.lstsq(x_a, x_b, rcond=None)[0]
        method = "least squares" if rows > cols and rank == cols else "minimum norm"
    result = _join(x, ub / ua)
    result = result.to(unit.strip()) if unit.strip() else result.to_reduced_units()
    residual = _join(x_b - x_a @ x, ub)
    largest = float(np.max(np.abs(residual.magnitude))) if residual.size else 0.0
    return LinearSolution(
        result,
        residual,
        cond,
        rank,
        method,
        {"Condition number": cond, "Rank": float(rank), f"Largest residual ({_name(ub)})": largest},
    )


@solve_linear.check
def _check_solve_linear(a: Any = None, unit: str = "") -> Any:
    if problem := _unit_problem(unit):
        return problem
    if a is None:
        return None
    try:
        arr, _ = _split(a, "A")
    except (TypeError, ValueError) as exc:
        return str(exc)
    cond = np.linalg.cond(arr.reshape(-1, 1) if arr.ndim == 1 else arr)
    if math.isfinite(cond) and cond > 1e12:
        return warning(f"A is nearly singular (condition number {cond:.3g})")
    return None


# --- eigenvalues ------------------------------------------------------------------------------


class Eigen(NamedTuple):
    values: Quantity
    vectors: NDArray[np.number]
    real: bool
    symmetric: bool
    summary: dict[str, float]


def _positive_definite(arr: NDArray[np.float64]) -> bool:
    try:
        np.linalg.cholesky(arr)
    except np.linalg.LinAlgError:
        return False
    return True


@node(category="Math/Matrices", title="Eigenvalues", sample=False)
def eigenvalues(
    a: Matrix,
    b: Annotated[
        Quantity | NDArray[np.floating] | None,
        Param(description="For the generalized problem A v = λ B v, such as K v = ω² M v"),
    ] = None,
    sort: Literal["ascending", "descending", "as computed"] = "ascending",
) -> Eigen:
    """The eigenvalues λ and eigenvectors v of A (A v = λ v), or of A against B
    (A v = λ B v). λ is in A's unit over B's. Column j of ``vectors`` belongs
    to value j, scaled to length 1.

    Symmetric matrices (Hermitian, when complex; with B positive definite)
    have real eigenvalues and are solved with the symmetric method; others may
    give complex pairs."""
    import scipy.linalg

    x, ua = _split(a, "A", square=True, vector_ok=False, complex_ok=True)
    y, ub = (
        (None, ureg().dimensionless) if b is None else _split(b, "B", square=True, complex_ok=True)
    )
    if y is not None and y.shape != x.shape:
        raise ValueError(f"A is {_shape(x)} but B is {_shape(y)}: they must be the same size")
    # the conjugate transpose: a complex symmetric matrix is not Hermitian, and eigh
    # would read only one triangle of it
    symmetric = bool(np.allclose(x, x.conj().T) and (y is None or np.allclose(y, y.conj().T)))
    if symmetric and (y is None or _positive_definite(y)):
        values, vectors = scipy.linalg.eigh(x, y)
        if y is not None:  # eigh scales to vᵀ B v = 1; here every vector has length 1
            vectors = vectors / np.linalg.norm(vectors, axis=0)
    else:
        values, vectors = scipy.linalg.eig(x, y)
        values = np.real_if_close(values, tol=1000)
        vectors = np.real_if_close(vectors, tol=1000)
    if sort != "as computed":
        order = np.lexsort((np.imag(values), np.real(values)))
        if sort == "descending":
            order = order[::-1]
        values, vectors = values[order], vectors[:, order]
    real = not np.iscomplexobj(values)
    summary: dict[str, float] = {}
    for i, v in enumerate(values, 1):
        if real:
            summary[f"λ{i}"] = float(v)
        else:
            summary[f"λ{i} (real)"] = float(np.real(v))
            summary[f"λ{i} (imaginary)"] = float(np.imag(v))
    return Eigen(_join(values, ua / ub), vectors, real, symmetric, summary)


class Modes(NamedTuple):
    frequency: Hertz
    angular_frequency: RadPerSecond
    shapes: NDArray[np.float64]
    first: Hertz
    plot: Figure
    summary: dict[str, float]


def _per_second_squared(ku: Any, mu: Any) -> float:
    """The factor that turns stiffness over mass into 1/s², or a clear error."""
    ratio = ureg().Quantity(1.0, ku / mu)
    if ratio.dimensionless:  # plain numbers: consistent SI units assumed
        return float(ratio.to("").magnitude)
    try:
        return float(ratio.to("1/s^2").magnitude)
    except Exception:
        raise ValueError(
            f"Stiffness over mass must be 1/time² (such as N/m over kg); "
            f"here it is {_name(ku)} over {_name(mu)}"
        ) from None


def _scale_shapes(vectors: NDArray[np.float64], normalise: str) -> NDArray[np.float64]:
    shapes = np.array(vectors, dtype=np.float64)
    for j in range(shapes.shape[1]):
        v = shapes[:, j]
        size = np.abs(v)
        # the first of the largest components, allowing for rounding between equal ones
        i = int(np.argmax(size >= size.max() * (1 - 1e-9)))
        shapes[:, j] = v / v[i] if normalise == "largest = 1" else v * np.sign(v[i])
    return shapes


@node(category="Math/Matrices", title="Natural Frequencies", sample=False)
def natural_frequencies(
    stiffness: Matrix,
    mass: Matrix,
    normalise: Annotated[
        Literal["largest = 1", "mass"],
        Param(description="Scale each mode shape so its largest entry is 1, or so φᵀ M φ = 1"),
    ] = "largest = 1",
) -> Modes:
    """The natural frequencies and mode shapes of an undamped system of
    masses and springs, from its stiffness matrix K and mass matrix M: the
    solutions of K φ = ω² M φ. Frequencies are in ascending order, and column
    j of ``shapes`` is mode j, the way the masses move at that frequency.

    K and M must be symmetric, and M positive definite. A mode with ω² ≈ 0 is
    a rigid-body motion (0 Hz), and a negative ω² (an unstable structure)
    gives NaN."""
    import scipy.linalg

    k, ku = _split(stiffness, "stiffness", square=True, vector_ok=False)
    m, mu = _split(mass, "mass", square=True)
    if m.ndim == 1:
        m = np.diag(m)
    if m.shape != k.shape:
        raise ValueError(f"stiffness is {_shape(k)} but mass is {_shape(m)}: they must match")
    if not np.allclose(k, k.T):
        raise ValueError("The stiffness matrix must be symmetric")
    if not np.allclose(m, m.T) or not _positive_definite(m):
        raise ValueError("The mass matrix must be symmetric and positive definite")
    lam, vectors = scipy.linalg.eigh(k * _per_second_squared(ku, mu), m)
    tiny = 1e-9 * max(float(np.max(np.abs(lam))), 1e-300)
    omega = np.where(np.abs(lam) <= tiny, 0.0, np.sqrt(np.where(lam > 0, lam, np.nan)))
    freq = omega / (2 * math.pi)
    shapes = _scale_shapes(vectors, normalise)

    fig = Figure(figsize=(6.0, 3.4), layout="constrained")
    ax = fig.add_subplot()
    dof = np.arange(1, len(freq) + 1)
    for j, f in enumerate(freq):
        ax.plot(dof, shapes[:, j], "o-", lw=2, label=f"mode {j + 1}, {f:.3g} Hz")
    ax.axhline(0.0, color="0.6", lw=1)
    ax.set(title="Mode shapes", xlabel="degree of freedom", ylabel="amplitude", xticks=dof)
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")

    reg = ureg()
    summary: dict[str, float] = {}
    for j, (f, w) in enumerate(zip(freq, omega, strict=True), 1):
        summary[f"f{j} (Hz)"] = float(f)
        summary[f"ω{j} (rad/s)"] = float(w)
    return Modes(
        reg.Quantity(freq, "Hz"),
        reg.Quantity(omega, "rad/s"),
        shapes,
        reg.Quantity(float(freq[0]), "Hz"),
        fig,
        summary,
    )


@natural_frequencies.check
def _check_natural_frequencies(stiffness: Any = None, mass: Any = None) -> Any:
    if stiffness is None or mass is None:
        return None
    try:
        _, ku = _split(stiffness, "stiffness")
        _, mu = _split(mass, "mass")
        _per_second_squared(ku, mu)
    except (TypeError, ValueError) as exc:
        return str(exc)
    if (ku / mu).dimensionless:
        return warning("Plain numbers: stiffness and mass are assumed to be in consistent SI units")
    return None
