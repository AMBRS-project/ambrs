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
* A working set of aerosol models, built using [ambuilder](https://github.com/AMBRS-project/ambuilder) (or whatever method you prefer)

We recommend using the `ambrs` framework inside its own [Python virtual environment](https://docs.python.org/3/library/venv.html). You can install its dependencies within a virtual
environment by running

```
pip install -r requirements.txt
```

within the top-level directory of this repository.

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

Only one of CAMP and MOSAIC may be enabled.
