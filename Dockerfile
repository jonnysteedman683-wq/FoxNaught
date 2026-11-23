FROM python:3.12-slim

RUN useradd --create-home --shell /usr/sbin/nologin foxnaught

WORKDIR /app
COPY foxnaught/ foxnaught/

ENV PYTHONUNBUFFERED=1 \
    PORT=8080
EXPOSE 8080

USER foxnaught

CMD ["python", "-m", "foxnaught", "--serve", "--host", "0.0.0.0", "--port", "8080"]
