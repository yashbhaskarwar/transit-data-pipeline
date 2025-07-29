FROM apache/airflow:2.7.0

USER airflow

RUN pip install --no-cache-dir \
    scikit-learn \
    xgboost \
    pandas \
    numpy \
    psycopg2-binary \
    python-dotenv

USER airflow
