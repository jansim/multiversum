# multiversum

[![PyPI](https://img.shields.io/pypi/v/multiversum.svg)](https://pypi.org/project/multiversum/)
[![Tests](https://github.com/jansim/multiversum/actions/workflows/test.yml/badge.svg)](https://github.com/jansim/multiversum/actions/workflows/test.yml)
[![Changelog](https://img.shields.io/github/v/release/jansim/multiversum?include_prereleases&label=changelog)](https://github.com/jansim/multiversum/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/jansim/multiversum/blob/main/LICENSE)

<p align="center">
  <img alt="multiversum logo" src="assets/logo.svg" width="60%" align="center">
</p>

`multiversum` is a package designed to make it easy to conduct multiverse analyses in Python. The package is intended to seemlessly integrate into a normal analysis or ML workflow and can also be added to an existing pipeline.

### Features

- **Simple** 🧩: Built with the goal of being as simple as possible to integrate into existing workflows.
- **Parallel** 👯: Different universes are automatically evaluated in parallel.
- **Notebooks** 📓: Analyses can be written as Jupyter notebooks.
- **Play/Pause** ⏯️: Interrupt and then continue a multiverse analysis where you left it.

## Installation

Install this library using `pip`:
```bash
pip install multiversum
```

## Usage

The package always works with two different files: The `multiverse.toml` ✨️, specifying the different dimensions (and their options) and the `universe.ipynb` ⭐️ containing the actual analysis code. The universe file is then evaluated (in parallel) using different dimension-combinations, by running `python -m multiversum`.

An example using a machine learning workflow in scikit-learn can be found [here](../examples/scikit-learn--simple/).