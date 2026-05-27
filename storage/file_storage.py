"""File storage for metadata and downloaded attachments."""

import io
import json
import logging
import os
import re
from pathlib import Path

from googleapiclient.http import MediaIoBaseDownload


class FileStorage:
    """Handles metadata JSON and Drive attachment downloads."""

    def __init__(self, base_dir: Path, download_dir: Path, data_dir: Path):
        self.base_dir = Path(base_dir)
        self.download_dir = Path(download_dir)
        self.data_dir = Path(data_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, name: str) -> str:
        name = name.strip()
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        safe = safe.strip("._")
        return safe[:180] if safe else "file"

    def save_metadata(
        self, item_type: str, course_id: str, item_id: str, record: dict, raw_payload=None
    ) -> Path:
        """Persist structured metadata as JSON for Hermes consumption."""
        target_dir = self.data_dir / item_type
        target_dir.mkdir(parents=True, exist_ok=True)
        file_name = self._safe_filename(f"{course_id}_{item_id}") + ".json"
        payload = dict(record)
        if raw_payload is not None:
            payload["raw_payload"] = raw_payload
        path = target_dir / file_name
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        return path

    def download_attachments(
        self,
        materials: list,
        item_type: str,
        course_id: str,
        item_id: str,
        drive_service,
    ) -> list:
        """Download Drive attachments and return their relative paths."""
        if not materials:
            return []
        if drive_service is None:
            logging.warning("Drive service missing, skipping downloads.")
            return []

        attachment_paths = []
        target_dir = self.download_dir / item_type / course_id / item_id
        target_dir.mkdir(parents=True, exist_ok=True)

        for material in materials:
            drive_wrapper = material.get("driveFile") or {}
            drive_file = drive_wrapper.get("driveFile") or {}
            file_id = drive_file.get("id")
            if not file_id:
                continue

            info = self._get_drive_file_info(drive_service, file_id, drive_file)
            title = info.get("name") or file_id
            file_name = self._safe_filename(title)
            export_mime, export_ext = self._get_export_format(info.get("mimeType"))
            if export_ext and not file_name.lower().endswith(export_ext):
                file_name = f"{file_name}{export_ext}"
            target_path = target_dir / file_name

            if not target_path.exists():
                try:
                    self._download_drive_file(
                        drive_service,
                        file_id,
                        target_path,
                        export_mime=export_mime,
                    )
                except Exception:
                    logging.exception("Failed to download Drive file %s", file_id)
                    continue

            relative_path = os.path.relpath(target_path, self.base_dir)
            attachment_paths.append(relative_path)

        return attachment_paths

    def _get_drive_file_info(
        self, drive_service, file_id: str, drive_file: dict
    ) -> dict:
        info = {
            "name": drive_file.get("title") or drive_file.get("name"),
            "mimeType": drive_file.get("mimeType"),
        }
        if info["name"] and info["mimeType"]:
            return info

        try:
            response = (
                drive_service.files()
                .get(fileId=file_id, fields="name,mimeType")
                .execute()
            )
            info["name"] = info["name"] or response.get("name")
            info["mimeType"] = info["mimeType"] or response.get("mimeType")
        except Exception:
            logging.exception("Failed to fetch Drive metadata for %s", file_id)
        return info

    def _get_export_format(self, mime_type: str | None) -> tuple[str | None, str | None]:
        if not mime_type:
            return None, None

        exports = {
            "application/vnd.google-apps.document": (
                "application/pdf",
                ".pdf",
            ),
            "application/vnd.google-apps.spreadsheet": (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".xlsx",
            ),
            "application/vnd.google-apps.presentation": (
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ".pptx",
            ),
            "application/vnd.google-apps.drawing": (
                "image/png",
                ".png",
            ),
        }
        return exports.get(mime_type, (None, None))

    def _download_drive_file(
        self,
        drive_service,
        file_id: str,
        target_path: Path,
        export_mime: str | None = None,
    ) -> None:
        if export_mime:
            request = drive_service.files().export_media(
                fileId=file_id, mimeType=export_mime
            )
        else:
            request = drive_service.files().get_media(fileId=file_id)
        with io.FileIO(target_path, "wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    logging.debug(
                        "Downloading %s: %.0f%%",
                        target_path.name,
                        status.progress() * 100,
                    )
