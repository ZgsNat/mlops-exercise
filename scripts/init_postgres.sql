-- Initialize isolated database for Airflow
SELECT 'CREATE DATABASE airflow_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow_db')\gexec

GRANT ALL PRIVILEGES ON DATABASE airflow_db TO mlops;
