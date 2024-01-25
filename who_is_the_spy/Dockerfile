FROM python:bullseye

RUN pip install leaf-playground==0.4.1

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python"]