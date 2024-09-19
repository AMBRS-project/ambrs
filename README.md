# ambrs

![Tests](https://github.com/AMBRS-project/ambrs/actions/workflows/tests.yml/badge.svg)

[![Coverage](https://codecov.io/gh/AMBRS-project/ambrs/graph/badge.svg?token=HF1V8JOZFJ)](https://codecov.io/gh/AMBRS-project/ambrs)
The AMBRS Project aims to advance the state of science in aerosol model
development by providing a simple framework for systematically comparing the
parameterizations in different model implementations. In this repository you'll
find an `ambrs` Python module that provides a workflow for running and comparing
the results of a set of curated aerosol box models. The framework also lets you
hook up your own box model for comparison with the curated set.

The framework is still in its infancy, so please reach out to someone on the
project if you're interested in participating.

## Supported Aerosol Box Models

All of the box models supported by AMBRS are forked under the [AMBRS-Project](https://github.com/AMBRS-project) GitHub organization. The box models we currently support are

* [PartMC](https://github.com/AMBRS-project/partmc)
* [MAM4](https://github.com/AMBRS-project/MAM_box_model)

AMBRS provides a [CMake](https://cmake.org/)-based [automated tool](https://github.com/AMBRS-project/ambuilder)
for configuring and building each of these aerosol models and their dependencies.

## System Requirements

To use the `ambrs` Python module, you need

* Python 3.12 or greater
* A working set of aerosol models, built using [ambuilder](https://github.com/AMBRS-project/ambuilder) (or whatever method you prefer)

We recommend using the `ambrs` framework inside its own [Python virtual environment](https://docs.python.org/3/library/venv.html). You can install its dependencies within a virtual
environment by running

```
pip install -r requirements.txt
```

within the top-level directory of this repository.
