#!/usr/bin/env python3
import os
import re
import io
import csv
import tempfile
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import fitz  # PyMuPDF
import keras_ocr
from google.cloud import storage
from google.oauth2 import service_account


import sys, numpy as np
print(">>> PY:", sys.executable)
print(">>> NumPy:", np.__version__)
assert np.__version__.startswith("1."), "Wrong NumPy at runtime (must be 1.x)"

# ---- PyMuPDF sanity (must run before using fitz) ----
import sys, importlib
print(">>> PY:", sys.executable)
fitz = importlib.import_module("fitz")
print(">>> fitz module:", fitz.__file__)
print(">>> has fitz.open:", hasattr(fitz, "open"))
if not hasattr(fitz, "open"):
    raise RuntimeError(
        "Wrong 'fitz' module on sys.path. It must be PyMuPDF. "
        "See fix instructions."
    )
# -----------------------------------------------------


# ---------- CONFIG (your defaults) ----------
PROJECT_ID   = "mptinc-playground"
BUCKET_NAME  = "gcs_document_bucket"
PREFIX       = "12-09-2025 samples/"
KEY_PATH     = "/home/adrian/PycharmProjects/KYC-document-pipeline/moniepoint-document-verification-service-playground/key.json"

# Where to save the token CSV locally (and optionally upload to GCS after)
OUT_CSV      = "kyc_tokens_tensorflow.csv"
UPLOAD_CSV_TO_GCS = False
OUT_CSV_GCS_PATH  = f"{PREFIX.rstrip('/')}/reports/{OUT_CSV}"

# File types to process
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
PDF_EXTS = {".pdf"}

# Parse filenames like "DRIVER_LICENSE_0123-4567-...pdf"
FNAME_PATTERN = re.compile(r"^(?P<doc>[A-Za-z_]+)_(?P<ref>[0-9A-Za-z-]+)\.(?P<ext>[A-Za-z0-9]+)$")

# ---------- Helpers ----------
def parse_doc_and_ref(filename: str) -> Tuple[str, str]:
    """
    Extract DOCUMENT_TYPE and reference from a filename like:
      BANK_STATEMENT_0ce25a...jpg
    Returns (doc_type, ref) or ("","") if not matched.
    """
    base = Path(filename).name
    m = FNAME_PATTERN.match(base)
    if not m:
        return "", ""
    return m.group("doc").upper(), m.group("ref")

def get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_file(KEY_PATH)
    return storage.Client(project=PROJECT_ID, credentials=creds)

def list_gcs_files(bucket: storage.Bucket, prefix: str) -> List[str]:
    uris = []
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        ext = Path(blob.name).suffix.lower()
        if ext in IMG_EXTS or ext in PDF_EXTS:
            uris.append(f"gs://{bucket.name}/{blob.name}")
    return uris

import io, mimetypes, math
from pathlib import Path
import numpy as np
import cv2
import fitz  # PyMuPDF
from PIL import Image

MAX_DIM = 2200  # cap largest side to control RAM/CPU

def sniff_mime_from_bytes(data: bytes, name_hint: str | None = None) -> str:
    # Try magic signatures first (optional dependency)
    try:
        import magic
        m = magic.Magic(mime=True)
        return m.from_buffer(data)
    except Exception:
        pass
    # Fallback to simple sniffing
    if data.startswith(b"%PDF"):
        return "application/pdf"
    # Guess from filename if provided
    if name_hint:
        mt, _ = mimetypes.guess_type(name_hint)
        if mt:
            return mt
    # Last resort
    return "application/octet-stream"

def download_bytes(storage_client, gcs_uri: str) -> tuple[bytes, str]:
    assert gcs_uri.startswith("gs://")
    _, _, rest = gcs_uri.partition("gs://")
    bucket_name, _, blob_path = rest.partition("/")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    data = blob.download_as_bytes()
    return data, Path(blob_path).name

def downscale_keep_ar(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def cv2_imread_from_bytes(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return img

def pdf_pages_to_images_from_bytes(pdf_bytes: bytes, dpi: int = 200):
    import importlib
    fitz = importlib.import_module("fitz")
    if not hasattr(fitz, "open"):
        raise RuntimeError("PyMuPDF not available: 'fitz.open' missing (wrong module shadowing?).")

    imgs = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            from PIL import Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            imgs.append(img)
    finally:
        doc.close()
    return imgs



def poly_to_xyxy_norm(box: np.ndarray, width: int, height: int) -> Tuple[float, float, float, float]:
    """
    Convert a 4-point polygon (keras-ocr format: 4x2 array [[x1,y1],...]) to normalized (xmin,ymin,xmax,ymax).
    """
    xs = box[:, 0]
    ys = box[:, 1]
    xmin = float(xs.min()) / width
    xmax = float(xs.max()) / width
    ymin = float(ys.min()) / height
    ymax = float(ys.max()) / height
    return xmin, ymin, xmax, ymax

# ---------- Main OCR Pipeline ----------
@dataclass
class OCRResult:
    file_uri: str
    page: int
    token: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    doc_type: str
    reference: str

def run_keras_ocr_on_image(image: Image.Image) -> List[Tuple[str, np.ndarray]]:
    """
    Runs keras-ocr on a PIL Image and returns list of (text, box) predictions.
    box is a 4x2 ndarray of pixel coordinates (float).
    """
    # keras-ocr expects numpy array in RGB shape (H,W,3)
    return pipeline.recognize([np.array(image)])[0]  # [(text, box), ...]

def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def run_keras_ocr_on_cv_image(cv_img_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    # keras-ocr expects RGB np.array
    rgb = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB)
    return pipeline.recognize([rgb])[0]

def process_single_uri(storage_client, gcs_uri: str) -> list[OCRResult]:
    data, name = download_bytes(storage_client, gcs_uri)
    mime = sniff_mime_from_bytes(data, name_hint=name)
    rows: list[OCRResult] = []

    doc_type, reference = parse_doc_and_ref(Path(gcs_uri).name)

    if mime == "application/pdf" or name.lower().endswith(".pdf"):
        # PDF path
        try:
            pil_pages = pdf_pages_to_images_from_bytes(data, dpi=200)
        except Exception as e:
            raise RuntimeError(f"PDF render failed: {e}") from e

        for page_idx, pil_img in enumerate(pil_pages, start=1):
            cv_img = pil_to_cv(pil_img)
            cv_img = downscale_keep_ar(cv_img, MAX_DIM)
            preds = run_keras_ocr_on_cv_image(cv_img)
            h, w = cv_img.shape[:2]
            for text, box in preds:
                text = (text or "").strip()
                if not text:
                    continue
                xmin, ymin, xmax, ymax = poly_to_xyxy_norm(np.array(box), w, h)
                rows.append(OCRResult(
                    file_uri=gcs_uri, page=page_idx, token=text,
                    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                    doc_type=doc_type, reference=reference
                ))
        return rows

    # Image path
    cv_img = cv2_imread_from_bytes(data)
    if cv_img is None:
        raise RuntimeError("Unable to decode image bytes (corrupt or unsupported format).")
    cv_img = downscale_keep_ar(cv_img, MAX_DIM)
    preds = run_keras_ocr_on_cv_image(cv_img)
    h, w = cv_img.shape[:2]
    for text, box in preds:
        text = (text or "").strip()
        if not text:
            continue
        xmin, ymin, xmax, ymax = poly_to_xyxy_norm(np.array(box), w, h)
        rows.append(OCRResult(
            file_uri=gcs_uri, page=1, token=text,
            xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
            doc_type=doc_type, reference=reference
        ))
    return rows


def write_rows_to_csv(rows: List[OCRResult], out_csv: str):
    out_dir = Path(out_csv).parent
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_uri", "page", "token", "xmin", "ymin", "xmax", "ymax", "doc_type", "reference"])
        for r in rows:
            w.writerow([r.file_uri, r.page, r.token, r.xmin, r.ymin, r.xmax, r.ymax, r.doc_type, r.reference])

def maybe_upload_csv(storage_client: storage.Client, local_path: str, bucket_name: str, gcs_path: str):
    if not UPLOAD_CSV_TO_GCS:
        return
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded report to gs://{bucket_name}/{gcs_path}")

def main():
    # Auth + clients
    creds = service_account.Credentials.from_service_account_file(KEY_PATH)
    storage_client = storage.Client(project=PROJECT_ID, credentials=creds)

    # Build keras-ocr pipeline once (downloads detector+recognizer weights on first run)
    global pipeline
    pipeline = keras_ocr.pipeline.Pipeline()  # uses TensorFlow under the hood

    bucket = storage_client.bucket(BUCKET_NAME)
    uris = list_gcs_files(bucket, PREFIX)
    if not uris:
        print(f"No images/PDFs found under gs://{BUCKET_NAME}/{PREFIX}")
        return
    print(f"Found {len(uris)} documents")

    all_rows: List[OCRResult] = []
    for i, uri in enumerate(uris, start=1):
        print(f"[{i}/{len(uris)}] {uri}")
        try:
            rows = process_single_uri(storage_client, uri)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  ERROR processing {uri}: {e}")

    # Save CSV
    write_rows_to_csv(all_rows, OUT_CSV)
    print(f"\nWrote: {Path(OUT_CSV).resolve()}  (rows: {len(all_rows)})")

    maybe_upload_csv(storage_client, OUT_CSV, BUCKET_NAME, OUT_CSV_GCS_PATH)
    print("Done.")

if __name__ == "__main__":
    main()