# Customer Churn & Retention Intelligence: Streamlit app image.
# Python 3.12 matches the environment that trained and saved the model.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CHURN_PROJECT_ROOT=/app \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# 1. Libraries first. This slow layer is reused until a requirements file changes.
COPY requirements-app.txt requirements-lock.txt ./
RUN python -m pip install -r requirements-app.txt -c requirements-lock.txt

# 2. Our package, the app, the config and the trained artifacts.
COPY pyproject.toml config.yaml ./
COPY src ./src
RUN python -m pip install --no-deps .
COPY .streamlit ./.streamlit
COPY app ./app
COPY artifacts ./artifacts

# 3. Run as an unprivileged user; only the log folder is writable.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app/logs
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).read().strip() == b'ok' else 1)"

CMD ["python", "-m", "streamlit", "run", "app/main.py"]