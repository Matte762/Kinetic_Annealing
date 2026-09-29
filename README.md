# Kinetic Annealing

A numerical study of simulated annealing from a kinetic and statistical-mechanics perspective.

The repository combines a short written report with a reproducible one-dimensional Python implementation. The experiments connect the Metropolis simulated-annealing algorithm to fixed-temperature Boltzmann-Gibbs equilibrium and track convergence through a coarse-grained relative-entropy estimate.

## Contents

- `Kinetic_Annealing_Report.pdf` - project report and mathematical discussion.
- `Simulated_Annealing/experiments.ipynb` - reproducible numerical experiments.
- `Simulated_Annealing/functions/` - annealing, ensemble, entropy, and plotting routines.
- `Simulated_Annealing/helpers/` - validation and Metropolis acceptance helpers.
- `Simulated_Annealing/figures/` and `Pictures/` - generated figures used by the project.

## Run the experiments

Create a Python environment, install the dependencies, and launch Jupyter from the implementation directory:

```bash
python -m pip install -r requirements.txt
cd Simulated_Annealing
jupyter lab experiments.ipynb
```

The code requires Python 3.10 or later.

## Reference

This project follows Lorenzo Pareschi, *Optimization by linear kinetic equations and mean-field Langevin dynamics* (2024), [arXiv:2401.05553](https://arxiv.org/abs/2401.05553).

The third-party paper is linked rather than redistributed in this repository.

## License

No open-source license has been assigned. Copyright remains with the repository author unless a license is added later.
