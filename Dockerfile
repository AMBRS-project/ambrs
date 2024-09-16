# Here we set up our curated set of box models

FROM fedora:41

RUN dnf -y update \
    && dnf -y install \
        less \
        tmux \
        gcc-gfortran \
        make \
        netcdf-fortran-devel \
        cmake \
        python3.12 \
    && dnf clean all

# copy box model config files into place
COPY config /config/

# build PartMC
RUN git clone https://github.com/AMBRS-Project/partmc/archive/refs/heads/master.zip /partmc \
    && mkdir -p /partmc/build \
    && cd /partmc/build \
    && cmake -D CMAKE_BUILD_TYPE=release \
             -D CMAKE_C_FLAGS_RELEASE="-O2 -g -Werror -Wall -Wextra" \
             -D CMAKE_Fortran_FLAGS_RELEASE="-O2 -g -Werror -fimplicit-none -Wall -Wextra -Wconversion -Wunderflow -Wimplicit-interface -Wno-compare-reals -Wno-unused -Wno-unused-parameter -Wno-unused-dummy-argument -fbounds-check" \
             /partmc \
    && make \
    && cp /partmc/build/src/partmc /bin

# build MAM4
RUN git clone https://github.com/AMBRS-project/MAM_box_model/archive/refs/heads/main.zip /mam4\
    && cp /config/mam4/CMakeLists.txt /mam4
    && cp /partmc/netcdf.cmake /mam4
    && mkdir -p /mam4/build \
    && cd /mam4/build \
    && cmake -D CMAKE_BUILD_TYPE=Release \
             -D CMAKE_C_FLAGS="-O2 -g -Wall" \
             -D CMAKE_Fortran_FLAGS="-O2 -g -Wall -ffree-form -ffree-line-length=0 -fallow-invalid-boz -fallow-argument-mismatch" \
             /mam4 \
    && make \
    && cp /mam4/build/mam4 /bin

# Install AMBRS
COPY / /
pip3 install -r requirements.txt
