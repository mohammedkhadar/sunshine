from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

STATE_BLOB = "sunshine/state.json"


class GCSStateStore:
    def __init__(self, bucket: str | None = None) -> None:
        self._bucket_name = bucket or os.getenv("SUNSHINE_STATE_BUCKET", "")
        if not self._bucket_name:
            raise RuntimeError("SUNSHINE_STATE_BUCKET env var required")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client()
        return self._client

    def get_last_seen_post_id(self) -> str | None:
        try:
            client = self._get_client()
            bucket = client.bucket(self._bucket_name)
            blob = bucket.blob(STATE_BLOB)
            if not blob.exists():
                return None
            data = json.loads(blob.download_as_string())
            return data.get("last_seen_post_id")
        except Exception as exc:
            logger.warning("Failed to read state: %s", exc)
            return None

    def set_last_seen_post_id(self, post_id: str) -> None:
        try:
            client = self._get_client()
            bucket = client.bucket(self._bucket_name)
            blob = bucket.blob(STATE_BLOB)
            blob.upload_from_string(
                json.dumps({"last_seen_post_id": post_id}),
                content_type="application/json",
            )
        except Exception as exc:
            logger.warning("Failed to write state: %s", exc)
