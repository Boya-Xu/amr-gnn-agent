FROM python:3.11

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv
RUN uv sync --no-dev --frozen

COPY src/ ./src/

EXPOSE 8000

CMD ["uv", "run", "python", "src/api.py"]