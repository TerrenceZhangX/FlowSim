# Slurm node image — controller, compute, and REST API
#
# Based on Ubuntu 22.04 with Slurm 23.11 + munge + JWT support.
# Used by slurm-compose.yaml.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gosu \
    libhttp-parser-dev \
    libjson-c-dev \
    libjwt-dev \
    libmunge-dev \
    munge \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Slurm 23.11 from source (includes slurmrestd + JWT auth)
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
        --enable-slurmrestd && \
    make -j"$(nproc)" && \
    make install && \
    rm -rf /tmp/slurm-*

# Create required directories and users
RUN useradd -r -s /sbin/nologin slurm && \
    mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm && \
    chown slurm:slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm

# Slurm config — 2 compute nodes, 1 GPU each
COPY slurm.conf /etc/slurm/slurm.conf

# JWT key for REST API auth
RUN dd if=/dev/urandom bs=32 count=1 2>/dev/null | base64 > /etc/slurm/jwt_hs256.key && \
    chown slurm:slurm /etc/slurm/jwt_hs256.key && \
    chmod 0600 /etc/slurm/jwt_hs256.key

CMD ["bash"]
