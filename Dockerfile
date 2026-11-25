FROM python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b
WORKDIR /app
COPY requirements.lock pyproject.toml VERSION README.md LICENSE NOTICE ./
COPY src/ ./src/
COPY data/ ./data/
COPY docs/project/DATA_CARD.md docs/project/MODEL_CARD.md ./docs/project/
COPY examples/ ./examples/
RUN python -m pip install --no-cache-dir -r requirements.lock \
    && python -m pip install --no-cache-dir --no-deps -e .
RUN useradd --create-home runner && mkdir -p /app/runs && chown runner:runner /app/runs
USER runner
ENTRYPOINT ["yk-evidence"]
CMD ["--help"]
