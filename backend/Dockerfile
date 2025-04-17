FROM python:3.10-slim

WORKDIR /app

COPY /backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/model
RUN mkdir -p /app/data
RUN mkdir -p /app/backend

COPY backend /app/backend
COPY model/car_mpg_model.pkl /app/model/car_mpg_model.pkl
COPY data/cars.csv /app/data/cars.csv

CMD ["python", "-m", "uvicorn", "backend.car_api:app", "--host", "0.0.0.0", "--port", "8000"]