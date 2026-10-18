# Base image
FROM python:3.11-slim as build

# Packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc

ENV PATH="/venv/bin:$PATH"

COPY .docker/requirements.txt .
RUN python -m venv /venv \
    && . /venv/bin/activate \
    && /venv/bin/pip install -r requirements.txt --upgrade

# Main image
FROM python:3.11-slim

# Vulnerabilities
RUN echo "Vulnerabilities patching" \
    && echo "deb https://deb.debian.org/debian/ unstable main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb-src https://deb.debian.org/debian/ unstable main contrib non-free" >> /etc/apt/sources.list \
    && echo "APT::Default-Release "jessie";" \
    && apt-get update \    
    && apt-get upgrade -y \
    && apt-get install -y git gcc \
    && apt-get install -y --no-install-recommends --only-upgrade python3-wheel python3-pip redis e2fsprogs libdb5.3 git\
    && apt-get clean autoclean \
    && apt-get autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}/ \
    && rm -rf /var/lib/apt/lists/*

# Security
RUN echo "Configuring security" \
    && groupadd -g 999 python \
    && useradd -r -u 999 -g python python \
    && mkdir /usr/src/app && chown python:python /usr/src/app

# Regular user - scientist
RUN groupadd -g 1000 -o scientist \
    && useradd -m -u 1000 -g 1000 -o -s /bin/bash scientist \
    && chown -R scientist:scientist /usr/src/app && chmod 755 /usr/src/app

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:$PATH"
    
# Python environement
COPY --chown=python:python --from=build /venv /venv

WORKDIR /usr/src/app

# entrypoint
CMD [ "python", "./main.py" ]