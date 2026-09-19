FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY agenttrace ./agenttrace
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 agenttrace
USER agenttrace
ENTRYPOINT ["agenttrace"]
CMD ["--help"]

