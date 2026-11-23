FROM python:3.12-slim

WORKDIR /app
COPY foxnaught/ foxnaught/

ENV PYTHONUNBUFFERED=1 \
    PORT=8080
EXPOSE 8080

CMD ["python", "-m", "foxnaught", "--serve", "--host", "0.0.0.0", "--port", "8080"]
