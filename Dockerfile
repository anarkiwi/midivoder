FROM ubuntu:26.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -o Acquire::Retries=5 \
    && apt-get install -y --no-install-recommends -o Acquire::Retries=5 \
        python3 \
        python3-pip \
        fluidsynth \
        espeak-ng \
        fluid-soundfont-gm \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
ENV MIDIVODER_SOUNDFONT=/usr/share/sounds/sf2/FluidR3_GM.sf2 \
    MIDIVODER_ARTIFACTS=/app/artifacts \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2
WORKDIR /app

FROM base AS deps
COPY pyproject.toml ./
RUN python3 -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; print('\n'.join(p['dependencies'] + p['optional-dependencies']['test']))" > /tmp/req.txt \
    && pip install --no-cache-dir --break-system-packages -r /tmp/req.txt

FROM deps AS test
COPY midivoder ./midivoder
COPY tests ./tests
RUN pip install --no-cache-dir --break-system-packages --no-deps -e .
CMD ["python3", "-m", "pytest", "-n", "auto", "--cov=midivoder", "--cov-report=term-missing", "--cov-fail-under=85"]
