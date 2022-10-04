FROM python:3.8-slim-bullseye as base
RUN set -eux; \
	apt-get update --quiet && \
	apt-get install --yes --no-install-recommends openmpi-bin libopenmpi-dev wget gcc git && \
	rm -rf /var/lib/apt/lists/*
RUN wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_amd64 -O /usr/local/bin/yq && chmod +x /usr/local/bin/yq
RUN pip install mpi4py==3.1.3
RUN pip install git+https://github.com/gridtools/gt4py@b39a0f0b85038592e3fccabde56137bb9c60e231#egg=gt4py


FROM base as target
LABEL purpose="Image to run the shallow water demo"
COPY . /usr/local/src/shallow_water
RUN pip install /usr/local/src/shallow_water[all]
RUN pip cache purge
env OMPI_ALLOW_RUN_AS_ROOT=1
env OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
WORKDIR /usr/local/src/shallow_water
ENTRYPOINT ["scripts/run_shallow_water.sh"]
