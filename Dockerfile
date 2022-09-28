FROM vicamo/pyenv:bullseye as base
ARG PYTHON_VERSION=3.8.13
RUN set -eux; \
	apt-get update --quiet && \
	apt-get install --yes --no-install-recommends openmpi-bin libopenmpi-dev wget && \
	rm -rf /var/lib/apt/lists/*
RUN wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_amd64 -O /usr/local/bin/yq && chmod +x /usr/local/bin/yq
RUN git -C /opt/pyenv pull
RUN pyenv install ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}
RUN pip install mpi4py
RUN pip install git+https://github.com/gridtools/gt4py@c08f9f60e26a4734cac26479af4593e86c91cd06#egg=gt4py


FROM base as target
LABEL purpose="Image to run the shallow water demo"
COPY . /usr/local/src/shallow_water
RUN pip install /usr/local/src/shallow_water
env OMPI_ALLOW_RUN_AS_ROOT=1
env OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
WORKDIR /usr/local/src/shallow_water
