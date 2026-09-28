FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home --uid 10001 airouter && mkdir -p /data && chown airouter /data
USER airouter
EXPOSE 11112 8500
# The de-identification salt must be supplied at run time:  -e AIROUTER_DEID_SALT=<secret>
CMD ["python", "-m", "airouter", "serve", "--config", "config/router.docker.yaml"]
