# multiversum

[![PyPI](https://img.shields.io/pypi/v/multiversum.svg)](https://pypi.org/project/multiversum/)
[![Tests](https://github.com/jansim/multiversum/actions/workflows/test.yml/badge.svg)](https://github.com/jansim/multiversum/actions/workflows/test.yml)
[![Changelog](https://img.shields.io/github/v/release/jansim/multiversum?include_prereleases&label=changelog)](https://github.com/jansim/multiversum/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/jansim/multiversum/blob/main/LICENSE)

`multiversum` is a package designed to make it easy to conduct multiverse analyses in Python. The package is intended to seemlessly integrate into a normal analysis or ML workflow and can also be added to an existing pipeline.

## Installation

Install this library using `pip`:
```bash
pip install multiversum
```

## Usage

The package always works with two different files: The `multiversum.toml` ✨️, specifying the different dimensions (and their options) and the `universe.ipynb` ⭐️ containing the actual analysis code. The universe file is then evaluated (in parallel) using different dimension-combinations.

An example using a machine learning workflow in scikit-learn can be found [here](./examples/scikit-learn--simple/).

## Development

To contribute to this library, first checkout the code. Then create a new virtual environment:

```bash
cd multiversum
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:

```bash
python -m pip install -e '.[test]'
```

To run the tests:

```bash
python -m pytest
```

### Formatting

Ruff is used for formatting and linting. Formatting can be automatically checked / applied wherever possible via `ruff check . --fix && ruff format`.
