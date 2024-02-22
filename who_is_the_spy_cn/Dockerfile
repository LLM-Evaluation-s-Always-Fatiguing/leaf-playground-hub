FROM python:bullseye

RUN pip install leaf-playground==0.6.0

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python"]