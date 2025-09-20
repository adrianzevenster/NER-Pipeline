#!/usr/bin/env python3

# First, install the required packages:
# !pip install google-cloud-documentai google-cloud-storage google-auth pandas

import os
from typing import List, Tuple, Dict, Any
import pandas as pd

from google.oauth2 import service_account
from google.cloud import documentai as docai
from google.cloud.documentai_v1 import types
from google.cloud import storage

# --- Config (yours) ---
# PROJECT_ID = "mptinc-playground"
# LOCATION = "eu"  # Your processors are in EU region as shown in screenshot
# PROCESSOR_ID = os.environ.get("DOC_OCR_PROCESSOR_ID", "vertex-pg-ai-ocr-processor")  # Use actual processor name from screenshot
# CREDENTIALS = "/home/adrian/PycharmProjects/KYC-document-pipeline/moniepoint-document-verification-service-playground/key.json"
#
# BUCKET = "gcs_document_bucket"
# PREFIX = "12-09-2025 samples/"

PROJECT_ID = "adg-delivery-moniepoint"
LOCATION = "eu"
PROCESSOR_ID = "c22f270a59d3af82"
CREDENTIALS="/home/adrian/PycharmProjects/KYC-document-pipeline/moniepoint-document-verification-service-playground/ProcessorTraining/.gcp/adg-documentai-sa-key.json"
BUCKET= "adg-delivery-moniepoint-docs-bucket-001"
PREFIX= "12-09-2025 samples/"

# Output (local file; you can upload it to GCS after if you want)
OUT_CSV = "kyc_tokens.csv"

# File types we'll process
VALID_EXTS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def get_mime_type(filename: str) -> str:
    """Get MIME type based on file extension."""
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


def list_gcs_uris(bucket: str, prefix: str, storage_client: storage.Client) -> List[Tuple[str, str]]:
    """List all valid document files in the GCS bucket with given prefix."""
    uris = []
    for blob in storage_client.list_blobs(bucket, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        name_lower = blob.name.lower()
        if any(name_lower.endswith(ext) for ext in VALID_EXTS):
            uri = f"gs://{bucket}/{blob.name}"
            mime_type = get_mime_type(blob.name)
            uris.append((uri, mime_type))
    return uris


def get_text(doc: types.Document, segment: types.Document.TextAnchor) -> str:
    """Reconstruct text for a token/paragraph/line using text segments."""
    if not segment or not segment.text_segments:
        return ""
    out = []
    text = doc.text or ""
    for seg in segment.text_segments:
        start = int(seg.start_index) if seg.start_index is not None else 0
        end = int(seg.end_index) if seg.end_index is not None else 0
        out.append(text[start:end])
    return "".join(out)


def norm_bbox(bbox) -> Tuple[float, float, float, float]:
    """
    Convert polygon (4 vertices, normalized) to (xmin, ymin, xmax, ymax), all in [0,1].
    """
    if not bbox or not bbox.normalized_vertices:
        return 0.0, 0.0, 0.0, 0.0

    xs = [v.x for v in bbox.normalized_vertices]
    ys = [v.y for v in bbox.normalized_vertices]
    return min(xs), min(ys), max(xs), max(ys)


def tokens_from_doc(doc: types.Document, gcs_uri: str) -> List[Dict[str, Any]]:
    """
    Extract all tokens (words) with their page, confidence, bbox.
    """
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

            rows.append(
                dict(
                    file_uri=gcs_uri,
                    page=page_idx,
                    token=text,
                    confidence=conf,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                )
            )
    return rows


def main():
    print("=== Google Cloud Document AI Configuration ===")
    print(f"Project ID: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    print(f"Processor ID: {PROCESSOR_ID}")
    print("=" * 50)

    if not PROCESSOR_ID:
        raise RuntimeError(
            "Please set your OCR processor id:\n"
            "  export DOC_OCR_PROCESSOR_ID=<your-processor-id>\n"
            "  or set it directly in the script"
        )

    # Check if credentials file exists
    if not os.path.exists(CREDENTIALS):
        raise RuntimeError(f"Credentials file not found: {CREDENTIALS}")

    # Auth
    creds = service_account.Credentials.from_service_account_file(CREDENTIALS)
    storage_client = storage.Client(project=PROJECT_ID, credentials=creds)

    # For EU regions, use the EU endpoint
    client_options = {"api_endpoint": "eu-documentai.googleapis.com"}

    docai_client = docai.DocumentProcessorServiceClient(
        credentials=creds,
        client_options=client_options
    )

    processor_name = docai_client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)
    print(f"Processor path: {processor_name}")

    # Test the processor first
    try:
        # Try to get processor info to verify it exists
        processor_info = docai_client.get_processor(name=processor_name)
        print(f"Found processor: {processor_info.display_name} (Type: {processor_info.type_})")
    except Exception as e:
        print(f"ERROR: Cannot access processor. Details: {e}")
        print("\nTroubleshooting steps:")
        print("1. Verify your processor ID is correct")
        print("2. Check that the location matches where your processor is deployed")
        print("3. Common EU regions: eu-west1, eu-west2, eu-west3, eu-west4, eu-central1, europe-west1")
        print("4. Run this command to list your processors:")
        print(f"   gcloud ai document-ai processors list --location={LOCATION}")
        return

    # Gather files to process
    try:
        uris = list_gcs_uris(BUCKET, PREFIX, storage_client)
    except Exception as e:
        print(f"Error accessing GCS bucket: {e}")
        return

    if not uris:
        print(f"No PDFs/images found under gs://{BUCKET}/{PREFIX}")
        return
    print(f"Found {len(uris)} files to OCR")

    all_rows: List[Dict[str, Any]] = []

    for idx, (uri, mime_type) in enumerate(uris, start=1):
        print(f"[{idx}/{len(uris)}] Processing {uri}")

        try:
            # Create the request with proper MIME type
            request = types.ProcessRequest(
                name=processor_name,
                gcs_document=types.GcsDocument(gcs_uri=uri, mime_type=mime_type),
                skip_human_review=True,
            )

            # Process the document
            result = docai_client.process_document(request=request)
            doc = result.document

            # Extract tokens
            rows = tokens_from_doc(doc, uri)
            all_rows.extend(rows)
            print(f"  → Extracted {len(rows)} tokens")

        except Exception as e:
            print(f"  → Error processing {uri}: {e}")
            continue

    if not all_rows:
        print("No tokens extracted from any documents.")
        return

    # Build dataframe and save
    df = pd.DataFrame(all_rows, columns=[
        "file_uri", "page", "token", "confidence", "xmin", "ymin", "xmax", "ymax"
    ])
    df.sort_values(["file_uri", "page"], inplace=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nWrote token CSV: {os.path.abspath(OUT_CSV)}")
    print(f"Total tokens extracted: {len(df)}")
    print("Columns: file_uri,page,token,confidence,xmin,ymin,xmax,ymax")
    print("\nNext steps:")
    print(" - Use this CSV to build span-level labels (NER) or region labels (layout).")
    print(" - If you want Label Studio JSON for bounding-box labeling, say the word and I'll emit it.")


if __name__ == "__main__":
    main()