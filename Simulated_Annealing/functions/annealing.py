from collections.abc import Callable, Sequence

import numpy as np

from helpers.helpers import acceptance_rule, as_integer, check_annealing_inputs


def simulated_annealing(
    fun: Callable[[float], float],
    x0: float,
    t0: float = 100.0,
    n_iter: int = 20000,
    seed: int | None = 762,
) -> tuple[float, float]:
    """Run simulated annealing and return the best x and objective value.

    Uses logarithmic cooling, ``temp = t0 / log(n + 2)``, following the
    convergence literature of Geman and Geman (1984) and Hajek (1988). Proposals
    use Gaussian random-walk increments, ``x_prop = x + sqrt(temp) * N(0, 1)``,
    accepted with the Metropolis rule from Metropolis et al. (1953). The overall
    optimization framing follows Kirkpatrick, Gelatt, and Vecchi (1983).
    """
    x0, t0, n_iter, seed = check_annealing_inputs(fun, x0, t0, n_iter, seed)
    rng = np.random.default_rng(seed=seed)

    x = x0
    value = float(fun(x))
    best_x = x
    best_value = value

    for n in range(n_iter):
        temp = t0 / np.log(n + 2)

        sigma = np.sqrt(temp)
        x_prop = x + sigma * rng.normal()
        value_prop = float(fun(x_prop))

        x, value = acceptance_rule(x, value, x_prop, value_prop, temp, rng)

        if value < best_value:
            best_x = x
            best_value = value

    return best_x, best_value


def empirical_fixed_temperature_distribution(
    fun: Callable[[float], float],
    x0: float,
    t0: float = 100.0,
    n_initial_cooling_steps: int = 5,
    total_runs: int = 10_000,
    stopping_iteration: int = 1_000,
    seed: int | None = 762,
) -> tuple[np.ndarray, float]:
    """Estimate the fixed-temperature empirical distribution."""
    x0, t0, stopping_iteration, seed = check_annealing_inputs(
        fun,
        x0,
        t0,
        stopping_iteration,
        seed,
        count_name="stopping_iteration",
    )
    n_initial_cooling_steps = as_integer(
        n_initial_cooling_steps, "n_initial_cooling_steps", minimum=0
    )
    total_runs = as_integer(total_runs, "total_runs", minimum=1)
    fixed_temp = t0 / np.log(n_initial_cooling_steps + 2)

    rng = np.random.default_rng(seed=seed)
    samples = np.empty(total_runs)
    fixed_sigma = np.sqrt(fixed_temp)

    for run in range(total_runs):
        x = x0
        value = float(fun(x))

        for n in range(n_initial_cooling_steps):
            temp = t0 / np.log(n + 2)
            sigma = np.sqrt(temp)
            x_prop = x + sigma * rng.normal()
            value_prop = float(fun(x_prop))
            x, value = acceptance_rule(x, value, x_prop, value_prop, temp, rng)

        for _ in range(stopping_iteration):
            x_prop = x + fixed_sigma * rng.normal()
            value_prop = float(fun(x_prop))
            x, value = acceptance_rule(x, value, x_prop, value_prop, fixed_temp, rng)

        samples[run] = x

    return samples, fixed_temp


def fixed_temperature_ensemble(
    fun: Callable[[float], float],
    x0: float,
    temperature: float,
    observation_times: Sequence[int],
    total_runs: int = 5_000,
    seed: int | None = 762,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample independent fixed-temperature trajectories at selected times.

    Every trajectory starts from ``x0`` and uses the same Gaussian Metropolis
    kernel as the annealing algorithm, but with a constant temperature. The
    returned sample array has shape ``(len(observation_times), total_runs)``.
    """
    x0, temperature, _, seed = check_annealing_inputs(
        fun, x0, temperature, 0, seed
    )
    total_runs = as_integer(total_runs, "total_runs", minimum=1)

    try:
        times = np.asarray(
            [
                as_integer(time, "observation time", minimum=0)
                for time in observation_times
            ],
            dtype=int,
        )
    except TypeError as exc:
        raise TypeError("observation_times must be a sequence of integers.") from exc

    if times.size == 0:
        raise ValueError("observation_times must not be empty.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("observation_times must be strictly increasing.")

    rng = np.random.default_rng(seed=seed)
    samples = np.empty((times.size, total_runs))
    proposal_scale = np.sqrt(temperature)
    final_time = int(times[-1])

    for run in range(total_runs):
        x = x0
        value = float(fun(x))
        observation_index = 0

        if times[0] == 0:
            samples[0, run] = x
            observation_index = 1

        for step in range(1, final_time + 1):
            x_prop = x + proposal_scale * rng.normal()
            value_prop = float(fun(x_prop))
            x, value = acceptance_rule(
                x, value, x_prop, value_prop, temperature, rng
            )

            if observation_index < times.size and step == times[observation_index]:
                samples[observation_index, run] = x
                observation_index += 1

    return times, samples


def empirical_relative_entropy(
    samples: np.ndarray,
    fun: Callable[[float], float],
    temperature: float,
    bin_edges: np.ndarray,
    quadrature_points: int = 20_001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate empirical densities and their relative entropy to equilibrium.

    The samples must be arranged by time, with shape ``(n_times, n_runs)`` (a
    one-dimensional sample is also accepted). Both the empirical law and the
    Boltzmann--Gibbs law are coarse-grained over ``bin_edges``. Consequently,
    the returned entropy is the discrete histogram estimate

    ``sum_i p_i * log(p_i / q_i)``,

    where ``q_i`` is obtained by numerical quadrature of ``exp(-fun(x) / T)``.
    The calculation is conditional on the finite interval spanned by the bins.
    """
    if not callable(fun):
        raise TypeError("fun must be callable.")

    _, temperature, _, _ = check_annealing_inputs(fun, 0.0, temperature, 0, None)
    quadrature_points = as_integer(
        quadrature_points, "quadrature_points", minimum=2
    )

    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim == 1:
        sample_array = sample_array[np.newaxis, :]
    if sample_array.ndim != 2 or sample_array.shape[1] == 0:
        raise ValueError("samples must have shape (n_times, n_runs) with n_runs > 0.")
    if not np.all(np.isfinite(sample_array)):
        raise ValueError("samples must contain only finite values.")

    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must be a one-dimensional array of length >= 2.")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be finite and strictly increasing.")
    if np.any(sample_array < edges[0]) or np.any(sample_array > edges[-1]):
        raise ValueError("all samples must lie inside the interval spanned by bin_edges.")

    widths = np.diff(edges)
    points_per_bin = max(3, int(np.ceil(quadrature_points / widths.size)))
    relative_grid = np.linspace(0.0, 1.0, points_per_bin)
    integration_grid = edges[:-1, np.newaxis] + widths[:, np.newaxis] * relative_grid
    try:
        objective_values = np.asarray(fun(integration_grid), dtype=float)
    except (TypeError, ValueError):
        objective_values = np.asarray(
            [float(fun(point)) for point in integration_grid.ravel()]
        ).reshape(integration_grid.shape)

    if objective_values.ndim == 0:
        objective_values = np.full_like(integration_grid, objective_values)
    if objective_values.shape != integration_grid.shape:
        raise ValueError("fun must return one value for each quadrature point.")
    if not np.all(np.isfinite(objective_values)):
        raise ValueError("fun must be finite on the interval spanned by bin_edges.")

    unnormalized_density = np.exp(
        -(objective_values - objective_values.min()) / temperature
    )
    equilibrium_bin_mass = np.sum(
        0.5
        * (unnormalized_density[:, :-1] + unnormalized_density[:, 1:]),
        axis=1,
    )
    equilibrium_bin_mass *= widths / (points_per_bin - 1)
    equilibrium_bin_mass = np.maximum(
        equilibrium_bin_mass, np.finfo(float).tiny
    )
    equilibrium_bin_mass /= equilibrium_bin_mass.sum()

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    equilibrium_density = equilibrium_bin_mass / widths
    empirical_densities = np.empty((sample_array.shape[0], bin_centers.size))
    relative_entropies = np.empty(sample_array.shape[0])

    for index, time_samples in enumerate(sample_array):
        counts, _ = np.histogram(time_samples, bins=edges)
        empirical_mass = counts / counts.sum()
        empirical_densities[index] = empirical_mass / widths

        occupied = empirical_mass > 0
        relative_entropies[index] = np.sum(
            empirical_mass[occupied]
            * np.log(empirical_mass[occupied] / equilibrium_bin_mass[occupied])
        )

    return (
        bin_centers,
        empirical_densities,
        equilibrium_density,
        relative_entropies,
    )


def trace_simulated_annealing(
    fun: Callable[[float], float],
    x0: float,
    t0: float = 100.0,
    n_iter: int = 20000,
    seed: int | None = 762,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run simulated annealing and return trace data.

    This is the traced version of ``simulated_annealing``. It uses the same
    logarithmic cooling, Gaussian random-walk proposal, and Metropolis
    acceptance rule, but also records the accepted trajectory and temperatures.
    """
    x0, t0, n_iter, seed = check_annealing_inputs(fun, x0, t0, n_iter, seed)
    rng = np.random.default_rng(seed=seed)

    x = x0
    value = float(fun(x))

    trajectory = np.empty(n_iter + 1)
    trajectory[0] = x

    temperatures = np.empty(n_iter)

    best_x = x
    best_value = value

    for n in range(n_iter):
        temp = t0 / np.log(n + 2)
        temperatures[n] = temp

        sigma = np.sqrt(temp)
        x_prop = x + sigma * rng.normal()
        value_prop = float(fun(x_prop))

        x, value = acceptance_rule(x, value, x_prop, value_prop, temp, rng)
        trajectory[n + 1] = x

        if value < best_value:
            best_x = x
            best_value = value

    return best_x, best_value, trajectory, temperatures
