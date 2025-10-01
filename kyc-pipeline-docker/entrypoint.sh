#!/bin/bash

# Default to DocumentAI if no arg
PIPELINE=${1:-documentai}

case "$PIPELINE" in
  documentai)
    echo "Running DocumentAI Pipeline..."
    python documentai_token_extraction.py
    python NER_documentai.py
    ;;
  tensorflow)
    echo "Running TensorFlow Pipeline..."
    python tensorflow_token_extraction.py
    python NER_tensorflow.py --tokens "${OUT_CSV:-kyc_tokens_tensorflow.csv}"
    ;;
  upload)
    echo "Running GCS Upload..."
    python drive_to_gcs.py
    ;;
  *)
    echo "Invalid: $PIPELINE. Use: documentai, tensorflow, upload"
    exit 1
    ;;
esac