from google.auth.environment_vars import CREDENTIALS

#!/usr/bin/env python3
"""
Upload ALL files from a Google Drive folder to a Google Cloud Storage bucket
under the prefix "Phase2/". Handles both binary files (PDF, images, etc.) and
Google Workspace files (Docs/Sheets/Slides/Drawings) via export, including
Shared Drives.

Defaults:
  --project  adg-delivery-moniepoint
  --bucket   adg-delivery-moniepoint-docs-bucket-001
  --folder   19U0hhGHzpbZFuV9EqeiFyf76T31y3w7Z
  --prefix   Phase2
  recursive  True (descend into subfolders)

Auth:
  Use a Google Cloud service account JSON key with access to BOTH Drive and Storage.
  1) Share the Drive folder (or add the SA as a member of the Shared Drive) with the SA email (Viewer+).
  2) Grant the SA Storage permissions on the bucket (legacyBucketReader + objectAdmin or project-wide storage.admin).
  3) Set GOOGLE_APPLICATION_CREDENTIALS to the SA JSON, or pass --credentials /path/key.json.

Dependencies:
  pip install google-api-python-client google-auth google-auth-httplib2 \
              google-auth-oauthlib google-cloud-storage
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from google.cloud import storage
from google.api_core.exceptions import NotFound, Conflict

# ---- CONFIG ----
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]  # full scope, robust for Shared Drives + export

# Preferred export formats for Google Workspace mimeTypes
# (mimeType -> (export_mime, extension))
GOOGLE_EXPORTS: Dict[str, Tuple[str, str]] = {
    "application/vnd.google-apps.document": ("application/pdf", "pdf"),  # Docs -> PDF
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xlsx",
    ),  # Sheets -> XLSX
    "application/vnd.google-apps.presentation": ("application/pdf", "pdf"),  # Slides -> PDF
    "application/vnd.google-apps.drawing": ("image/png", "png"),  # Drawings -> PNG
}

FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"


@dataclass
class Ctx:
    drive: Any
    storage_client: storage.Client
    bucket: storage.Bucket
    prefix: str
    recursive: bool
    drive_id: Optional[str]  # Shared Drive id when applicable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload ALL Drive files to GCS under Phase2 prefix")
    p.add_argument("--project", default="mptinc-playground")
    p.add_argument("--bucket", default="gcs_document_bucket")
    p.add_argument("--folder", default="19U0hhGHzpbZFuV9EqeiFyf76T31y3w7Z", help="Drive folder ID")
    p.add_argument("--prefix", default="Phase2", help="Object key prefix in GCS")
    p.add_argument("--credentials", default=None, help="Path to service account JSON (optional)")
    p.add_argument("--no-recursive", dest="recursive", action="store_false", help="Do not descend into subfolders")
    args, _ = p.parse_known_args()  # Jupyter/Notebook-friendly
    if not hasattr(args, "recursive"):
        args.recursive = True
    return args


def load_credentials(json_path: Optional[str]):
    """
    Return credentials with DRIVE_SCOPES and a fresh access token.
    Prefer explicit service account JSON; fall back to ADC if needed.
    """
    if json_path:
        creds = service_account.Credentials.from_service_account_file(json_path, scopes=DRIVE_SCOPES)
    else:
        creds, _ = google.auth.default(scopes=DRIVE_SCOPES)

    if not creds.valid:
        creds.refresh(Request())
    return creds


def build_clients(creds, project: str):
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    storage_client = storage.Client(project=project, credentials=creds)
    return drive, storage_client


def assert_drive_access(drive, folder_id: str) -> Optional[str]:
    """
    Ensure we can see the folder; return driveId if it's on a Shared Drive (else None).
    """
    try:
        meta = drive.files().get(
            fileId=folder_id,
            fields="id,name,mimeType,driveId",
            supportsAllDrives=True,
        ).execute()
        print(f"Drive OK: '{meta['name']}' ({meta['mimeType']}), driveId={meta.get('driveId')}")
        return meta.get("driveId")  # None means it's in My Drive
    except HttpError as e:
        txt = (e.content or b"").decode("utf-8", "ignore")
        print("Drive access check failed:", txt, file=sys.stderr)
        raise


def ensure_bucket(storage_client: storage.Client, bucket_name: str) -> storage.Bucket:
    try:
        return storage_client.get_bucket(bucket_name)
    except NotFound:
        print(f"Bucket '{bucket_name}' not found; creating it...", flush=True)
        try:
            # Set location if you need a specific region, e.g., location="africa-south1"
            bucket = storage_client.create_bucket(bucket_name)
            return bucket
        except Conflict:
            # Was created concurrently
            return storage_client.get_bucket(bucket_name)


def sanitize_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", name)
    return (
        name.replace("\\", "_")
        .replace("/", "_")
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


def list_children(drive, folder_id: str, drive_id: Optional[str]):
    """
    Yield direct children of folder_id.
    If drive_id is set, restrict query to that Shared Drive.
    """
    q = f"'{folder_id}' in parents and trashed = false"
    page_token = None
    params = dict(
        q=q,
        spaces="drive",
        fields=("nextPageToken, files(id, name, mimeType, size, modifiedTime, "
                "shortcutDetails(targetId, targetMimeType))"),
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    )
    if drive_id:
        params["driveId"] = drive_id
        params["corpora"] = "drive"

    while True:
        if page_token:
            params["pageToken"] = page_token
        resp = drive.files().list(**params).execute()
        for f in resp.get("files", []):
            yield f
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def resolve_shortcut(file_obj: Dict[str, Any]) -> Dict[str, Any]:
    # For shortcuts, substitute the target id/mimeType but keep the visible name
    details = file_obj.get("shortcutDetails") or {}
    target_id = details.get("targetId")
    target_mime = details.get("targetMimeType")
    if target_id and target_mime:
        return {**file_obj, "id": target_id, "mimeType": target_mime}
    return file_obj


def export_google_file(drive, file_id: str, export_mime: str) -> bytes:
    request = drive.files().export_media(
        fileId=file_id,
        mimeType=export_mime,
        supportsAllDrives=True,
    )
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request, chunksize=1024 * 1024)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"  Export {pct}%", end="\r", flush=True)
    return buf.getvalue()


def download_binary_file(drive, file_id: str) -> bytes:
    request = drive.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request, chunksize=1024 * 1024)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"  Download {pct}%", end="\r", flush=True)
    return buf.getvalue()


def upload_bytes_to_gcs(bucket: storage.Bucket, blob_name: str, data: bytes, content_type: Optional[str]):
    blob = bucket.blob(blob_name)
    blob.cache_control = "no-store"
    blob.upload_from_string(data, content_type=content_type)


def gcs_key(prefix: str, path_parts: list[str]) -> str:
    # Join with '/', avoiding doubles
    tail = "/".join([p for p in path_parts if p])
    return f"{prefix}/{tail}" if prefix else tail


def process_folder(ctx: Ctx, folder_id: str, path: list[str]) -> Tuple[int, int]:
    """Return (uploaded, skipped)."""
    uploaded = 0
    skipped = 0

    for f in list_children(ctx.drive, folder_id, ctx.drive_id):
        f = resolve_shortcut(f) if f.get("mimeType") == SHORTCUT_MIME else f
        mime = f.get("mimeType")
        name = sanitize_name(f.get("name") or f.get("id"))
        file_id = f["id"]

        if mime == FOLDER_MIME:
            if ctx.recursive:
                print(f"[DIR] {name}")
                u, s = process_folder(ctx, file_id, path + [name])
                uploaded += u
                skipped += s
            else:
                print(f"Skipping subfolder (non-recursive): {name}")
                skipped += 1
            continue

        print(f"[FILE] {name} ({mime}) id={file_id}")

        try:
            # Google Workspace export
            if mime.startswith("application/vnd.google-apps"):
                if mime in GOOGLE_EXPORTS:
                    export_mime, ext = GOOGLE_EXPORTS[mime]
                    data = export_google_file(ctx.drive, file_id, export_mime)
                    key = gcs_key(ctx.prefix, path + [f"{name}.{ext}"])
                    upload_bytes_to_gcs(ctx.bucket, key, data, export_mime)
                    print(f" -> gs://{ctx.bucket.name}/{key}")
                    uploaded += 1
                else:
                    print(f"  WARN: Unsupported Google file type for export: {mime}. Skipping.")
                    skipped += 1
            else:
                # Binary download
                data = download_binary_file(ctx.drive, file_id)
                key = gcs_key(ctx.prefix, path + [name])
                upload_bytes_to_gcs(ctx.bucket, key, data, f.get("mimeType"))
                print(f" -> gs://{ctx.bucket.name}/{key}")
                uploaded += 1
        except Exception as e:
            skipped += 1
            print(f"  ERROR: {name}: {e}", file=sys.stderr)

    return uploaded, skipped


def main():
    args = parse_args()

    creds = load_credentials(args.credentials)
    print("Scopes:", getattr(creds, "scopes", None))
    print("Service account:", getattr(creds, "service_account_email", None))

    drive, storage_client = build_clients(creds, args.project)
    # Assert access and capture Shared Drive id (if any) BEFORE listing
    drive_id = assert_drive_access(drive, args.folder)
    bucket = ensure_bucket(storage_client, args.bucket)

    ctx = Ctx(
        drive=drive,
        storage_client=storage_client,
        bucket=bucket,
        prefix=args.prefix,
        recursive=args.recursive,
        drive_id=drive_id,
    )

    start = time.time()
    print(f"Starting upload from Drive folder {args.folder} (recursive={ctx.recursive}) → gs://{bucket.name}/{args.prefix}/")
    uploaded, skipped = process_folder(ctx, args.folder, path=[])
    dur = time.time() - start
    print(f"Done. Uploaded: {uploaded}, Skipped: {skipped}, Elapsed: {dur:.1f}s")


if __name__ == "__main__":
    main()
