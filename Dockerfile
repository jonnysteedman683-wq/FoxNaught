FROM python:3.12-slim

RUN useradd --create-home --shell /usr/sbin/nologin foxnaught

WORKDIR /app
COPY foxnaught/ foxnaught/

ENV PYTHONUNBUFFERED=1 \
    PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8080\")}/health', timeout=3)"]

USER foxnaught

CMD ["python", "-m", "foxnaught", "--serve", "--host", "0.0.0.0", "--port", "8080"]
