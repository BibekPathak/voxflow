FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PATH="/root/.local/bin:$PATH"

WORKDIR /srv/voxflow

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
COPY app ./app

RUN uv sync --frozen --no-dev --no-install-project

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
