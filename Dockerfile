FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10

RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y

RUN pip install --upgrade pip

WORKDIR /final_app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY src/ .

EXPOSE 5000

CMD ["python3", "-m" , "flask", "run", "--host=0.0.0.0"]
