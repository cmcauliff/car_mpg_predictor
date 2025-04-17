FROM python:3.10-slim

WORKDIR /app

COPY frontend/requirements.txt .
COPY frontend .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "car_mpg_app.py", "--server.port=8501", "--server.address=0.0.0.0"]