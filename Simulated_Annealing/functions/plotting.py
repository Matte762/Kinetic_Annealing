from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from helpers.helpers import as_finite_float, as_integer


def plot_optimization_result(
    fun: Callable[[float | np.ndarray], float | np.ndarray],
    best_x: float,
    best_value: float,
    x_min: float = -25.0,
    x_max: float = 25.0,
    n_points: int | None = None,
    margin: float = 0.05,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plot the objective function and the best minimum found."""
    if not callable(fun):
        raise TypeError("fun must be callable.")

    x_min = as_finite_float(x_min, "x_min")
    x_max = as_finite_float(x_max, "x_max")

    if x_min >= x_max:
        raise ValueError("x_min must be smaller than x_max.")

    margin = as_finite_float(margin, "margin")

    if margin < 0:
        raise ValueError("margin must be non-negative.")

    if n_points is not None:
        n_points = as_integer(n_points, "n_points", minimum=2)

    best_x = float(best_x)
    best_value = float(best_value)

    plot_min = x_min
    plot_max = x_max

    if np.isfinite(best_x):
        span = x_max - x_min
        padding = margin * span
        plot_min = min(plot_min, best_x - padding)
        plot_max = max(plot_max, best_x + padding)

    if n_points is None:
        plot_span = plot_max - plot_min
        if np.isfinite(plot_span):
            n_points = max(1000, int(30 * plot_span))
            n_points = min(n_points, 50000)
        else:
            n_points = 50000

    x_grid = np.linspace(plot_min, plot_max, n_points)
    y_grid = np.asarray(fun(x_grid), dtype=float)

    if y_grid.ndim == 0:
        y_grid = np.full_like(x_grid, y_grid)

    if y_grid.shape != x_grid.shape:
        raise ValueError("fun must return one value for each point in the plot grid.")

    fig, ax = plt.subplots()
    ax.plot(x_grid, y_grid, label="Objective function")

    if np.isfinite(best_x) and np.isfinite(best_value):
        ax.scatter(best_x, best_value, color="red", label="Best minimum found")
    elif np.isposinf(best_x) and np.isfinite(best_value):
        ax.axhline(best_value, color="red", linestyle="--", label="Best value at +inf")
    elif np.isneginf(best_x) and np.isfinite(best_value):
        ax.axhline(best_value, color="red", linestyle="--", label="Best value at -inf")

    ax.set_xlim(plot_min, plot_max)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Simulated annealing result")
    ax.legend()
    ax.grid(True)

    if show:
        plt.show()

    return fig, ax
