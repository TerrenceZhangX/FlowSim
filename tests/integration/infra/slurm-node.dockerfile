# Slurm node image — controller, compute, and REST API
#
# Based on flowsim-image so compute nodes have the full Python/sglang
# environment.  Slurm 23.11 is compiled on top with JWT + NVML GRES.
# Used by slurm-compose.yaml.

FROM flowsim-image:latest

ENV DEBIAN_FRONTEND=noninteractive

# Slurm build dependencies + munge
RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    libhttp-parser-dev \
    libjson-c-dev \
    libjwt-dev \
    libmunge-dev \
    munge \
    && rm -rf /var/lib/apt/lists/*

# Install Slurm 23.11 from source (slurmrestd + JWT auth + NVML GRES)
ARG SLURM_VERSION=23.11.10
RUN cd /tmp && \
    wget -q https://download.schedmd.com/slurm/slurm-${SLURM_VERSION}.tar.bz2 && \
    tar xjf slurm-${SLURM_VERSION}.tar.bz2 && \
    cd slurm-${SLURM_VERSION} && \
    ./configure \
        --prefix=/usr \
        --sysconfdir=/etc/slurm \
        --with-jwt \
        --with-http-parser \
        --with-json \
        --with-nvml \
        --enable-slurmrestd && \
    make -j"$(nproc)" && \
    make install && \
    rm -rf /tmp/slurm-*

# Create required directories and users
RUN useradd -r -s /sbin/nologin slurm 2>/dev/null || true && \
    mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm && \
    chown slurm:slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm

# Slurm config
COPY slurm.conf /etc/slurm/slurm.conf
COPY gres.conf /etc/slurm/gres.conf
COPY cgroup.conf /etc/slurm/cgroup.conf

# JWT key for REST API auth
RUN dd if=/dev/urandom bs=32 count=1 2>/dev/null | base64 > /etc/slurm/jwt_hs256.key && \
    chown slurm:slurm /etc/slurm/jwt_hs256.key && \
    chmod 0600 /etc/slurm/jwt_hs256.key

WORKDIR /flowsim
CMD ["bash"]
