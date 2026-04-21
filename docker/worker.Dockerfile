FROM python:3.12-slim

WORKDIR /app

COPY requirements-backend.txt /app/requirements-backend.txt
RUN pip install --no-cache-dir -r /app/requirements-backend.txt

COPY . /app

CMD ["python", "-m", "backend.workers.docking_worker"]
