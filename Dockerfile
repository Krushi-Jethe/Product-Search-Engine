FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL author="Jethe Krushi, krushi.jethe@gmail.com"
LABEL version="1.0"
LABEL description="Product Search Engine app with GPU support"

WORKDIR /search-engine

RUN apt-get update && \
    apt-get install -y gcc portaudio19-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app/app.py"]
