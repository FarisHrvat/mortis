# MORTIS in a container.
#
#   docker build -t mortis .
#   docker run --rm -v "$PWD:/work" mortis info
#   docker run --rm -v "$PWD:/work" mortis run analysis.yaml
#
# Two stages, because building the wheels needs a compiler and running them
# does not. Carrying gcc into the final image would roughly double the download
# for no benefit to anyone actually running an analysis.
#
# Pinned to Python 3.11 deliberately. 3.12 resolves pandas 3 and anndata 0.13,
# which are supported and tested but move faster than most people want under a
# container they are trusting with a result.

FROM python:3.11-slim AS build

# Only needed to compile wheels that have no prebuilt arm64 build yet. None of
# this survives into the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/

# A virtualenv so the whole install copies to the runtime stage as one layer,
# without pip, its cache, or the build tree tagging along.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[cli,harmony]"


FROM python:3.11-slim AS runtime

# libgomp is the one shared library the wheels genuinely need at run time —
# numba and scikit-learn link OpenMP. HDF5 comes bundled inside the h5py
# wheel, so there is no system hdf5 package to keep in step with it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash mortis

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER mortis
WORKDIR /work

# Fail the build rather than ship an image that cannot import itself.
RUN python -c "import mortis, mortis.cli; print('MORTIS', mortis.__version__, 'ok')"

ENTRYPOINT ["mortis"]
CMD ["--help"]
