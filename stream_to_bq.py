from main_modules import logger
import requests
import json

def stream_to_bq(row: dict, endpoint: str, dataset: str, table: str) -> str:
    """
    Streams data to BigQuery. Using the Cloud Function: stream_to_bigquery.
    """
    logger("Streaming to BigQuery.")
    bq_url = endpoint
    payload = json.dumps({
        "data": row,
        "output": {
            "dataset": dataset,
            "table": table
        }
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", bq_url, headers=headers, data=payload)
    logger(response.text)
    return response.text