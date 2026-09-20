FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH="/work:${PYTHONPATH}" \
    AIRFLOW_HOME="/work/airflow"

WORKDIR /work

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv binary from official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy uv dependency definitions
COPY pyproject.toml uv.lock ./

# Install frozen dependencies into container's virtual environment with uv
RUN uv sync --frozen --no-dev --no-install-project

# Ensure virtualenv binaries are prioritized
ENV PATH="/work/.venv/bin:$PATH"

# Copy application and pipeline source files
COPY src/ ./src/
COPY app/ ./app/
COPY dags/ ./dags/
COPY scripts/ ./scripts/

CMD ["python", "--version"]
