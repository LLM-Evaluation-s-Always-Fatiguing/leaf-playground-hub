FROM python:bullseye

RUN pip install leaf-playground==0.4.0.dev0

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python"]