FROM python:3.8-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements-backend.txt

CMD ["python", "-c", "print('DeePathNet adapter container ready')"]
