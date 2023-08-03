FROM python:3.9.12

RUN apt-get update && apt-get install -y portaudio19-dev

WORKDIR /docker-app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
