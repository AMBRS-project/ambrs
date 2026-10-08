# ambrs

![Tests](https://github.com/AMBRS-project/ambrs/actions/workflows/tests.yml/badge.svg) [![Coverage](https://codecov.io/gh/AMBRS-project/ambrs/graph/badge.svg?token=HF1V8JOZFJ)](https://codecov.io/gh/AMBRS-project/ambrs)

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
* The [`uv`](https://docs.astral.sh/uv/) package manager
* A working set of aerosol models, built using [ambuilder](https://github.com/AMBRS-project/ambuilder) (or whatever method you prefer)

To clone and build `ambrs` in a virutal environment, run:

```sh
git clone https://github.com/AMBRS-project/ambrs.git
cd ambrs
uv sync
source .venv/bin/activate
uv pip install -e .
```

## Building Aerosol Models

AMBRS uses [ambuilder](https://github.com/AMBRS-project/ambuilder) to configure
and build the supported aerosol box models. From an ambuilder checkout, run:

```sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<prefix> -G "Unix Makefiles" -DENABLE_CAMP=OFF -DENABLE_MOSAIC=ON
cd build
make
make install
```

`<prefix>` is the installation directory where ambuilder will place model
executables under `bin`.

Additional options can be passed to CMake using the `-D` flag:

* `CMAKE_BUILD_TYPE={Debug,Release}`: builds debuggable or optimized versions of libraries and aerosol box models (default: `Release`)
* `CMAKE_C_COMPILER=/path/to/c-compiler`: sets the C compiler used to build libraries and/or aerosol box models
* `CMAKE_Fortran_COMPILER=/path/to/fortran-compiler`: sets the Fortran compiler used to build libraries and/or aerosol box models
* `CMAKE_INSTALL_PREFIX=/path/to/install`: sets the top-level directory under which supported aerosol box models are installed, with executables in a `bin` subdirectory
* `ENABLE_CAMP={ON,OFF}`: enables support for CAMP chemistry in relevant aerosol box models (default: `OFF`)
* `ENABLE_MOSAIC={ON,OFF}`: enables support for MOSAIC in relevant aerosol box models, using a branch maintained by the PartMC team (default: `OFF`)

Only one of CAMP and MOSAIC may be enabled. `ambrs` expects `/path/to/install/bin` to be on your `PATH` or in the `AMBRS_MODEL_DIR`
environment variable:
```sh
export AMBRS_MODEL_DIR=/path/to/install/bin
```

## Running Tutorial Notebooks

Tutorial notebooks are located in `tutorial/notebooks`. Prior to running the notebooks, you will need to build
the aerosol models and make them available in your virtual environment.

First, clone `ambrs` and `ambuilder`:
```sh
git clone https://github.com/AMBRS-project/ambrs.git
git clone https://github.com/AMBRS-project/ambuilder.git
```

Next, create the virtaul environment for `ambrs`:
```sh
cd ambrs
uv sync
source .venv/bin/activate
```

Then, build the aerosol models and install them in the virtual environment:
```sh
cmake \
  -S ../ambuilder \
  -B build \
  -D CMAKE_INSTALL_PREFIX="$VIRTUAL_ENV" \
  -G "Unix Makefiles" \
  -D ENABLE_CAMP=OFF \
  -D ENABLE_MOSAIC=ON
cmake --build build
cmake --install build
```

Finally, build the `ambrs` package with Jupyter dependencies included, and register the kernel:
```sh
uv pip install -e .[notebooks]
uv run python -m ipykernel install --user --name=python3
```

You can now run the tutorial notebooks using the registered Jupyter kernel.
