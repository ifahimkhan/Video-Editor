"""Media formats accepted by the Open dialog and file drops."""
from pathlib import Path

MEDIA_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mp3", ".wav", ".m4a")


def supported_media_file(path: str) -> bool:
    source = Path(path)
    return source.suffix.lower() in MEDIA_EXTENSIONS and source.is_file()
