from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

# Make our app module importable inside the Airflow container
sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def scrape_task(**context):
    """
    Task 1: Scrape earnings call transcripts and save to disk.
    We separate this from embedding so if embedding fails,
    we don't re-scrape everything unnecessarily.
    """
    from app.ingest import fetch_transcripts
    fetch_transcripts(output_dir='/opt/airflow/data/transcripts')
    print("Scraping complete")


def embed_task(**context):
    """
    Task 2: Chunk, embed, and store transcripts in pgvector.
    Only runs if scrape_task succeeds.
    """
    from app.rag.embedder import embed_and_store
    embed_and_store(transcript_dir='/opt/airflow/data/transcripts')
    print("Embedding complete")


def validate_task(**context):
    """
    Task 3: Verify data made it into the database correctly.
    Acts as a quality gate — if chunk counts look wrong,
    the DAG run is marked failed so we get alerted.

    Why a validation task? In production pipelines silent data
    quality failures are worse than loud ones. If embedding
    silently stored 0 chunks we'd never know without this check.
    This is a core MLOps pattern called data validation gating.
    """
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv()

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM document_chunks;")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()

    print(f"Total chunks in database: {count}")

    if count < 10:
        raise ValueError(f"Validation failed: only {count} chunks found, expected at least 10")

    print("Validation passed")


with DAG(
    dag_id='earnings_call_ingestion',
    default_args=default_args,
    description='Scrape, embed, and store earnings call transcripts quarterly',
    schedule_interval='0 9 1 2,5,8,11 *',  # 9am on the 1st of Feb, May, Aug, Nov
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['rag', 'earnings', 'ingestion']
) as dag:

    scrape = PythonOperator(
        task_id='scrape_transcripts',
        python_callable=scrape_task,
    )

    embed = PythonOperator(
        task_id='embed_and_store',
        python_callable=embed_task,
    )

    validate = PythonOperator(
        task_id='validate_chunks',
        python_callable=validate_task,
    )

    # Define task dependencies — this is the DAG structure
    # scrape must succeed before embed runs
    # embed must succeed before validate runs
    scrape >> embed >> validate