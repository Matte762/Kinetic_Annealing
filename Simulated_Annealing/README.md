# Simulated Annealing

Small Python project for one-dimensional simulated annealing.

## Structure

```text
.
├── functions/
│   ├── annealing.py
│   └── plotting.py
├── helpers/
│   └── helpers.py
├── experiments.ipynb
├── .gitignore
└── README.md
```

## Usage

Open `experiments.ipynb` and run the cells. The notebook contains the numerical
experiments: one section runs the optimizer and one estimates the
Boltzmann-Gibbs distribution by running many independent fixed-temperature runs
for a long time. A final fixed-temperature experiment follows an ensemble at
several times and estimates its relative entropy to Boltzmann-Gibbs equilibrium.
The notebook displays plots inline and does not save image files.

The notebook defines its objective and plotting windows directly:

```python
def objective(x):
    return ...

x_min = -25.0
x_max = 25.0
```

For plotting, the objective should work with NumPy arrays. Reusable package
logic stays under `functions/` and `helpers/`.

## Algorithm Choices

This project uses a deliberately simple one-dimensional simulated annealing
scheme. At iteration `n`, the temperature is

```python
temperature = t0 / log(n + 2)
```

This logarithmic schedule is slow but standard in the convergence theory for
simulated annealing. Geman and Geman (1984) used logarithmic cooling in the
context of stochastic relaxation and Gibbs distributions, and Hajek (1988)
gave sharp convergence conditions for schedules of the form `c / log(1 + t)`.

Candidate points are generated with Gaussian random-walk increments:

```python
x_prop = x + sqrt(temperature) * normal_random_value
```

The `sqrt(temperature)` scale makes proposals larger at high temperature and
smaller as the system cools. Proposed uphill moves are accepted using the
Metropolis probability `exp(-delta_f / temperature)`, tracing back to
Metropolis et al. (1953). The use of simulated annealing for optimization is
classically associated with Kirkpatrick, Gelatt, and Vecchi (1983).

These references motivate the cooling schedule, temperature-dependent proposal
scale, and acceptance rule used here. The code is not trying to model kinetic
theory.

## Library Functions

```python
simulated_annealing(
    fun: Callable[[float], float],
    x0: float,
    t0: float = 100.0,
    n_iter: int = 20000,
    seed: int | None = 762,
) -> tuple[float, float]
```

Returns:

```python
(best_x, best_value)
```

```python
trace_simulated_annealing(
    fun: Callable[[float], float],
    x0: float,
    t0: float = 100.0,
    n_iter: int = 20000,
    seed: int | None = 762,
) -> tuple[float, float, np.ndarray, np.ndarray]
```

Returns:

```python
(best_x, best_value, trajectory, temperatures)
```

```python
empirical_fixed_temperature_distribution(
    fun: Callable[[float], float],
    x0: float,
    t0: float = 100.0,
    n_initial_cooling_steps: int = 5,
    total_runs: int = 10_000,
    stopping_iteration: int = 1_000,
    seed: int | None = 762,
) -> tuple[np.ndarray, float]
```

Runs `total_runs` independent fixed-temperature experiments from the same
initial point `x0`. Each run first performs `n_initial_cooling_steps`
logarithmic-cooling updates, then fixes the temperature at
`t0 / log(n_initial_cooling_steps + 2)` and records the value after
`stopping_iteration` fixed-temperature updates. Larger `stopping_iteration`
estimates the long-time fixed-temperature distribution; smaller values estimate
the finite-time distribution from `x0`.

Returns:

```python
(samples, fixed_temp)
```

```python
fixed_temperature_ensemble(
    fun,
    x0,
    temperature,
    observation_times,
    total_runs=5_000,
    seed=762,
) -> tuple[np.ndarray, np.ndarray]
```

Runs independent Metropolis trajectories at a constant temperature and returns
the selected times together with a sample matrix of shape
`(len(observation_times), total_runs)`.

```python
empirical_relative_entropy(
    samples,
    fun,
    temperature,
    bin_edges,
    quadrature_points=20_001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Coarse-grains the empirical and Boltzmann-Gibbs laws on common histogram bins.
It returns the bin centers, empirical densities, equilibrium density, and the
discrete relative entropy at every sampled time. The finite bin interval must
contain all samples.

```python
plot_optimization_result(
    fun: Callable[[float | np.ndarray], float | np.ndarray],
    best_x: float,
    best_value: float,
    x_min: float = -25.0,
    x_max: float = 25.0,
    n_points: int | None = None,
    margin: float = 0.05,
    show: bool = True,
) -> tuple[Figure, Axes]
```

Produces the objective function plot over the requested window. If the best
finite point found is outside that window, the plot expands to include it with a
small margin. If `n_points` is `None`, the function chooses more grid points for
larger intervals, capped at 50,000. If the best point is reported as `+inf` or
`-inf`, the function draws a horizontal line at the finite best value instead of
trying to plot an infinite x-coordinate.

## Input Validation

Inputs are checked for the objective function, finite scalar bounds, positive
temperature, integer counts, and seed. Boolean values are rejected for integer
parameters even though Python treats `bool` as a subclass of `int`.
