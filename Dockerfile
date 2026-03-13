FROM registry.access.redhat.com/ubi8/python-311:latest AS test
USER root
RUN dnf -y upgrade && dnf -y install make curl \
    && dnf clean all \
    && python -m venv /venv \
    && mkdir /build
ENV VIRTUAL_ENV=/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
WORKDIR /build
COPY ./app ./app
COPY ./res ./res
COPY ./test ./test
COPY ./test_res ./test_res
COPY ./requirements.txt ./requirements.txt
COPY ./requirements-dev.txt ./requirements-dev.txt
COPY ./Makefile ./Makefile
COPY ./.flake8 ./.flake8
RUN "${VIRTUAL_ENV}/bin/pip" install --upgrade pip \
    && LIBRARY_PATH=/lib:/usr/lib /bin/sh -c "${VIRTUAL_ENV}/bin/pip install --no-cache-dir -r requirements.txt" \
    && "${VIRTUAL_ENV}/bin/python3" -m nltk.downloader -d /usr/share/nltk_data stopwords wordnet omw-1.4
RUN "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r requirements-dev.txt
RUN make test-all

FROM registry.access.redhat.com/ubi8/python-311:latest AS builder
USER root
RUN dnf -y upgrade && dnf -y install make curl \
    && dnf clean all \
    && python -m venv /venv \
    && mkdir /build
ENV VIRTUAL_ENV=/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
WORKDIR /build
COPY ./app ./app
COPY ./res ./res
COPY ./requirements.txt ./requirements.txt
COPY ./VERSION ./VERSION
COPY ./Makefile ./Makefile
RUN "${VIRTUAL_ENV}/bin/pip" install --upgrade pip \
    && "${VIRTUAL_ENV}/bin/pip" install --upgrade setuptools \
    && LIBRARY_PATH=/lib:/usr/lib /bin/sh -c "${VIRTUAL_ENV}/bin/pip install --no-cache-dir -r requirements.txt" \
    && "${VIRTUAL_ENV}/bin/python3" -m nltk.downloader -d /usr/share/nltk_data stopwords wordnet omw-1.4
ARG APP_VERSION=""
ARG RELEASE_MODE=false
ARG GITHUB_TOKEN
ARG HF_TOKEN
RUN if [ "$RELEASE_MODE" = "true" ]; then make release v=${APP_VERSION} githubtoken=${GITHUB_TOKEN}; else if [ "${APP_VERSION}" != "" ]; then make build-release v=${APP_VERSION}; fi ; fi
RUN HF_TOKEN=${HF_TOKEN} PYTHONPATH=/build "${VIRTUAL_ENV}/bin/python3" app/ml/bake_models.py
RUN mkdir /backend \
    && cp /build/VERSION /backend \
    && cp -r /build/app /backend/ \
    && cp -r /build/res /backend/

FROM registry.access.redhat.com/ubi8/python-311:latest
USER root
WORKDIR /backend/
COPY --from=builder /backend ./
COPY --from=builder /venv /venv
COPY --from=builder /usr/share/nltk_data /usr/share/nltk_data/

ENV VIRTUAL_ENV="/venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" PYTHONPATH=/backend

RUN dnf -y upgrade \
    && dnf -y install curl \
    && dnf clean all \
    && mkdir -p -m 0744 /backend/storage

# Start server
CMD ["/venv/bin/python", "app/main.py"]
HEALTHCHECK --interval=1m --timeout=5s --retries=2 CMD ["curl", "-s", "-f", "--show-error", "http://localhost:5001/"]
