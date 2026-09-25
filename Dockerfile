# moe-LARS in a container. Docker on macOS runs Linux, which has no Metal, so the MLX backend
# does not run here: the image serves the mock backend, or GGUF models on CPU through
# llama.cpp (build with --build-arg EXTRAS=llamacpp and mount the model).
#
#   docker build -t moelars .                                   # mock backend only
#   docker build -t moelars:llamacpp --build-arg EXTRAS=llamacpp .
#   docker run --rm -p 8600:8600 moelars
#   docker run --rm -p 8600:8600 -v "$PWD/models:/models:ro" moelars:llamacpp \
#       --backend llamacpp --model /models/<model>.gguf --template chatml

ARG PYTHON=3.12

FROM python:${PYTHON}-slim AS build
ARG EXTRAS=""
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
# A compiler and CMake only matter for llama-cpp-python, which builds llama.cpp from source.
RUN if [ -n "$EXTRAS" ]; then apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*; fi
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project $(for e in $EXTRAS; do printf -- '--extra %s ' "$e"; done)
COPY src ./src
RUN uv sync --frozen --no-dev $(for e in $EXTRAS; do printf -- '--extra %s ' "$e"; done)

FROM python:${PYTHON}-slim
# llama.cpp's CPU build links OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 moelars
COPY --from=build --chown=moelars /app /app
COPY --chown=moelars evals/samples /app/evals/samples
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER moelars
WORKDIR /app
EXPOSE 8600
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8600/healthz', timeout=4)"
ENTRYPOINT ["moelars", "serve", "--host", "0.0.0.0", "--port", "8600"]
CMD ["--backend", "mock"]
