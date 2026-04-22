FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 7860

CMD ["python", "app.py"]