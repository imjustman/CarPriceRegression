FROM python:3.10-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./App ./App
COPY ./Data ./Data
COPY ./Model ./Model
COPY ./Artifacts ./Artifacts
COPY config_prod.yml .

CMD sh -c "uvicorn App.app:app --host 0.0.0.0 --port 8003 & streamlit run App/main.py --server.port 8502 --server.address 0.0.0.0"