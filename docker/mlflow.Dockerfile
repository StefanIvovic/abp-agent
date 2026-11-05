# The official MLflow image doesn't include a Postgres driver; add it for the backend store.
# Its SQLAlchemy (2.1) maps postgresql:// to psycopg 3, so install that rather than psycopg2.
FROM ghcr.io/mlflow/mlflow:v3.17.2
RUN pip install --no-cache-dir "psycopg[binary]"
