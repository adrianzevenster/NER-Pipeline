#!/usr/bin/env python3

import os
from typing import List, Dict, Any
import pandas as pd
import datetime
import logging

from google.oauth2 import service_account
from google.cloud import documentai as docai
from google.cloud.documentai_v1 import types
from google.cloud import storage
from google.cloud import bigquery  # Add for BigQuery option

# Config from env for flexibility in Cloud Functions/Run
PROJECT_ID = os.environ.get("PROJECT_ID", "adg-delivery-moniepoint")
LOCATION = os.environ.get("LOCATION", "eu")
PROCESSOR_ID = os.environ.get("PROCESSOR_ID", "c22f270a59d3af82")
CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
INPUT_BUCKET = os.environ.get("INPUT_BUCKET", "adg-delivery-moniepoint-docs-bucket-001")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", INPUT_BUCKET)  # Same or separate
TOKENS_TABLE = os.environ.get("TOKENS_TABLE", "your_dataset.kyc_tokens")  # For BigQuery
USE_BIGQUERY = os.environ.get("USE_BIGQUERY", "true").lower() == "true"  # Toggle storage

VALID_EXTS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

logging.basicConfig(level=logging.INFO)

def get_mime_type(filename: str) -> str:
    ext = filename.lower().split('.')[-1]
    mime_map = {
        'pdf': 'application/pdf',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'tif': 'image/tiff',
        'tiff': 'image/tiff',
        'webp': 'image/webp'
    }
    return mime_map.get(ext, 'application/octet-stream')

def get_text(doc: types.Document, segment: types.Document.TextAnchor) -> str:
    if not segment or not segment.text_segments:
        return ""
    out = []
    text = doc.text or ""
    for seg in segment.text_segments:
        start = int(seg.start_index) if seg.start_index is not None else 0
        end = int(seg.end_index) if seg.end_index is not None else 0
        out.append(text[start:end])
    return "".join(out)

def norm_bbox(bbox) -> tuple:
    if not bbox or not bbox.normalized_vertices:
        return 0.0, 0.0, 0.0, 0.0
    xs = [v.x for v in bbox.normalized_vertices]
    ys = [v.y for v in bbox.normalized_vertices]
    return min(xs), min(ys), max(xs), max(ys)

def tokens_from_doc(doc: types.Document, gcs_uri: str) -> List[Dict[str, Any]]:
    rows = []
    for page_idx, page in enumerate(doc.pages, start=1):
        for token in page.tokens:
            if not token.layout or not token.layout.text_anchor:
                continue
            text = get_text(doc, token.layout.text_anchor).strip()
            if not text:
                continue
            conf = token.layout.confidence if token.layout.confidence else 0.0
            xmin, ymin, xmax, ymax = norm_bbox(token.layout.bounding_poly)
            rows.append({
                "file_uri": gcs_uri,
                "page": page_idx,
                "token": text,
                "confidence": conf,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "extracted_at": datetime.datetime.now().isoformat()
            })
    return rows

def save_tokens(rows: List[Dict[str, Any]]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if USE_BIGQUERY:
        bq_client = bigquery.Client(project=PROJECT_ID)
        job = bq_client.insert_rows_json(TOKENS_TABLE, rows)
        if job:
            logging.error(f"BigQuery insert errors: {job}")
        else:
            logging.info(f"Inserted {len(rows)} tokens to {TOKENS_TABLE}")
    else:
        # Append to GCS CSV
        storage_client = storage.Client(project=PROJECT_ID)
        blob = storage_client.bucket(OUTPUT_BUCKET).blob("kyc_tokensCumulative.csv")
        if blob.exists():
            existing_df = pd.read_csv(f"gs://{OUTPUT_BUCKET}/kyc_tokensCumulative.csv")
            df = pd.concat([existing_df, df])
        df.to_csv(f"gs://{OUTPUT_BUCKET}/kyc_tokensCumulative.csv", index=False)
        logging.info(f"Appended {len(rows)} tokens to GCS CSV")

def process_document(event, context):
    """Cloud Function entry point, triggered by GCS upload."""
    file = event['name']
    bucket = event['bucket']
    if bucket != INPUT_BUCKET or not any(file.lower().endswith(ext) for ext in VALID_EXTS):
        logging.info(f"Skipping non-document: {file}")
        return

    uri = f"gs://{bucket}/{file}"
    mime_type = get_mime_type(file)
    logging.info(f"Processing {uri}")

    creds = service_account.Credentials.from_service_account_file(CREDENTIALS) if CREDENTIALS else None
    client_options = {"api_endpoint": "eu-documentai.googleapis.com"}
    docai_client = docai.DocumentProcessorServiceClient(credentials=creds, client_options=client_options)
    processor_name = docai_client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

    # Use batch for async processing (better for automation)
    input_config = types.BatchDocumentsInputConfig(
        gcs_documents=types.GcsDocuments(documents=[types.GcsDocument(gcs_uri=uri, mime_type=mime_type)])
    )
    output_config = types.DocumentOutputConfig(
        gcs_output_config=types.DocumentOutputConfig.GcsOutputConfig(gcs_uri=f"gs://{OUTPUT_BUCKET}/processed/")
    )
    request = types.BatchProcessRequest(
        name=processor_name,
        input_documents=input_config,
        document_output_config=output_config,
        skip_human_review=True
    )
    operation = docai_client.batch_process_documents(request)
    operation.result()  # Wait for completion in function (or use Pub/Sub for long-running)

    # Fetch output JSON from GCS and extract
    storage_client = storage.Client(project=PROJECT_ID)
    prefix = f"processed/{os.path.basename(file)}/"  # Adjust based on output structure
    blobs = list(storage_client.list_blobs(OUTPUT_BUCKET, prefix=prefix))
    for blob in blobs:
        if blob.name.endswith('.json'):
            doc_json = types.Document.from_json(blob.download_as_string())
            rows = tokens_from_doc(doc_json, uri)
            save_tokens(rows)
            break

if __name__ == "__main__":
    # For local testing
    process_document({'name': 'test.pdf', 'bucket': INPUT_BUCKET}, None)