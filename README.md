# Multi-objective reinforcement learning with Signal Temporal Logic specifications

This repository implements a benchmark of tasks (`stlclash`) and experiments to compare single-objective and multi-objective RL methods for control with STL specifications.
It constitutes part of the work done by [Jon Goikoetxea](https://jongoiko.github.io/) during a research internship in [Verimag](https://www-verimag.imag.fr/?lang=en), under the supervision of Prof. [Alexandre Donzé](https://www-verimag.imag.fr/PEOPLE/donzeal/).


To run the experiments using [uv](https://docs.astral.sh/uv/), execute

```shell
uv run python -m scripts.benchmark_methods
```
This will generate a file `results.pickle` which can then be [read](https://docs.python.org/3/library/pickle.html) and analyzed.
The deserialized object is a list, with one dictionary per executed experiment.
Each dictionary contains information about the trained algorithm and its type (SO/MO), the task, and the best achieved success rates/robustnesses for the policy search algorithms.
Besides the best achieved metrics, the associated weights over sub-formulae are also recorded.