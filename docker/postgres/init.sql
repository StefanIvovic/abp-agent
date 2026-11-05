-- Runs once, the first time the postgres volume is created.
-- The app database (POSTGRES_DB) is created by the image; MLflow and Prefect get their own databases.
CREATE DATABASE mlflow;
CREATE DATABASE prefect;
