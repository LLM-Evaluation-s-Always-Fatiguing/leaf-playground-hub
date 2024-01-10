FROM python:bullseye

RUN pip install --extra-index-url https://prod-tanka-sg-nexus.aws.tankaapps.com/repository/pypi-ai-public/simple leaf-playground==0.3.0.dev0

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python"]