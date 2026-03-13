FROM python:3.11-slim-bookworm AS system-deps
USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /venv \
    && mkdir /build
WORKDIR /build

FROM system-deps AS python-deps
COPY ./requirements/ ./requirements/
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements/base.txt \
    && python -m nltk.downloader -d /usr/share/nltk_data stopwords wordnet omw-1.4

FROM python-deps AS model-download
COPY ./requirements/ ./requirements/
RUN pip install --no-cache-dir -r requirements/build.txt
COPY ./app ./app
COPY ./res ./res
COPY ./VERSION ./VERSION
COPY ./Makefile ./Makefile
ARG APP_VERSION=""
ARG RELEASE_MODE=false
ARG GITHUB_TOKEN
ARG HF_TOKEN
ARG BAKE_MODELS=false
RUN if [ "$RELEASE_MODE" = "true" ]; then make release v=${APP_VERSION} githubtoken=${GITHUB_TOKEN}; elif [ "${APP_VERSION}" != "" ]; then make build-release v=${APP_VERSION}; fi
RUN if [ "$BAKE_MODELS" = "true" ]; then HF_TOKEN=${HF_TOKEN} PYTHONPATH=/build python app/ml/bake_models.py; fi
RUN mkdir /backend \
    && cp /build/VERSION /backend \
    && cp -r /build/app /backend/ \
    && cp -r /build/res /backend/

FROM python-deps AS test
COPY ./requirements/ ./requirements/
RUN pip install --no-cache-dir -r requirements/test.txt
COPY ./app ./app
COPY ./res ./res
COPY ./test ./test
COPY ./test_res ./test_res
COPY ./Makefile ./Makefile
COPY ./.flake8 ./.flake8
RUN make test-all

FROM python:3.11-slim-bookworm AS runtime
USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV="/venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PYTHONPATH=/backend
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --uid 10001 appuser
WORKDIR /backend/
COPY --from=model-download --chown=appuser:appuser /backend ./
COPY --from=python-deps --chown=appuser:appuser /venv /venv
COPY --from=python-deps --chown=appuser:appuser /usr/share/nltk_data /usr/share/nltk_data/
RUN mkdir -p -m 0744 /backend/storage \
    && chown appuser:appuser /backend/storage
USER appuser

# Start server
CMD ["/venv/bin/python", "app/main.py"]
HEALTHCHECK --interval=1m --timeout=5s --retries=2 CMD ["curl", "-s", "-f", "--show-error", "http://localhost:5001/"]
