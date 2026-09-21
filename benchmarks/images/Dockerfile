# A benchmark image is built only from an immutable BASE_IMAGE value, for
# example python:3.12-slim@sha256:<64-hex-digest>.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /workspace
RUN python -m pip install --no-cache-dir pytest==9.0.2

ENTRYPOINT []
