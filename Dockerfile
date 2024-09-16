# Here we set up our curated set of box models

FROM fedora:41

# NOTE: we install Fedora's contourpy and matplotlib packages because the builds
# NOTE: for both of these fail to the tune of:
#  Run-time dependency python found: NO (tried pkgconfig, pkgconfig and sysconfig)
#  ../src/meson.build:5:10: ERROR: Python dependency not found
RUN dnf -y update \
    && dnf -y install \
        cmake \
        gcc-c++ \
        gcc-gfortran \
        git \
        less \
        make \
        netcdf-fortran-devel \
        openblas-devel \
        python3.12-devel \
        python3-contourpy \
        python3-matplotlib \
        python3-pip \
        tmux \
    && dnf clean all

# copy box model config files into place
COPY config /config/

# build PartMC
RUN git clone --depth 1 https://github.com/AMBRS-Project/partmc.git /partmc \
    && mkdir -p /partmc/build \
    && cd /partmc/build \
    && cmake -D CMAKE_BUILD_TYPE=release \
             -D CMAKE_C_FLAGS_RELEASE="-O2 -g -Werror -Wall -Wextra" \
             -D CMAKE_Fortran_FLAGS_RELEASE="-O2 -g -Werror -fimplicit-none -Wall -Wextra -Wconversion -Wunderflow -Wimplicit-interface -Wno-compare-reals -Wno-unused -Wno-unused-parameter -Wno-unused-dummy-argument -fbounds-check" \
             /partmc \
    && make \
    && cp /partmc/build/partmc /usr/local/bin

# build MAM4
RUN git clone --depth 1 https://github.com/AMBRS-project/MAM_box_model.git /mam4\
    && cp /config/mam4/CMakeLists.txt /mam4 \
    && cp /partmc/netcdf.cmake /mam4 \
    && mkdir -p /mam4/build \
    && cd /mam4/build \
    && cmake -D CMAKE_BUILD_TYPE=Release /mam4 \
    && make \
    && cp /mam4/build/mam4 /usr/local/bin

# install AMBRS
COPY / /
RUN pip3 install -r requirements.txt
