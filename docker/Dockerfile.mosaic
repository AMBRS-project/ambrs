FROM quay.io/jupyter/base-notebook:python-3.13

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        gcc \
        g++ \
        gfortran \
        git \
        libnetcdf-dev \
        libnetcdff-dev \
        make \
    && apt-get clean

# install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="${HOME}/.local/bin/:$PATH"

# get ambrs and ambuilder
COPY . ${HOME}/ambrs
RUN git clone https://github.com/AMBRS-project/ambuilder.git

# prepare the Jupyter kernel
RUN cd ambrs \
    && uv sync \
    && cmake \
        -S ${HOME}/ambuilder \
        -B build \
        -D CMAKE_INSTALL_PREFIX="${HOME}/ambrs/.venv" \
        -G "Unix Makefiles" \
        -D ENABLE_CAMP=OFF \
        -D ENABLE_MOSAIC=ON \
    && cmake --build build \
    && cmake --install build \
    && uv pip install -e .[notebooks] \
    && uv run python -m ipykernel install --user --name=ambrs-venv --display-name="Python (ambrs)" \
    && fix-permissions "${HOME}"

USER ${NB_UID}
WORKDIR ${HOME}/ambrs