**Named Entity Recognition (NER) Pipeline**

This repository provides a Named Entity Recognition (NER) pipeline designed to process and analyze text extracted from multiple document types.

The pipeline performs two main functions:

Text Extraction – Captures text content from structured and unstructured documents.

Entity Label Suggestion – Identifies and proposes candidate entity labels that are consistently present across each document type.

This approach is particularly useful for:

- Discovering common entity labels within document collections.

- Defining entity fields required for downstream data extraction tasks.

- Standardizing entity label usage across varied document sources.

docker run --rm \
-v /path/to/your/input/docs:/input-docs \  # If scripts need local inputs; otherwise GCS handles
-v /path/to/your/outputs:/outputs \       # Outputs (CSVs, reports, JSONs) here
-v /path/to/your/gcp-key.json:/app/gcp-key.json \  # GCP key
-e GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-key.json \
-e PROJECT_ID=adg-delivery-moniepoint \
-e LOCATION=eu \
-e PROCESSOR_ID=c22f270a59d3af82 \
-e BUCKET=adg-delivery-moniepoint-docs-bucket-001 \
-e PREFIX="12-09-2025 samples/" \
-e OUT_CSV=/outputs/kyc_tokens_documentai.csv \
-e TOKENS_CSV=/outputs/kyc_tokens_documentai.csv \  # For NER
kyc-pipeline documentai
