FROM python:3.7.13-buster

COPY api /api
COPY requirements.txt /requirements.txt
COPY spaceship_titanic /spaceship_titanic

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0
