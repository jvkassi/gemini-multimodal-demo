import os

EMBEDDING_MODEL = "gemini-embedding-2-preview"
SUPPORTED_FILE_TYPES = {
    "pdf": "application/pdf",
    "txt": "text/plain",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "mp4": "video/mp4",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
}
SUPPORTED_EXTENSIONS = list(SUPPORTED_FILE_TYPES.keys())
MAX_FILE_SIZE_MB = 100
POLLING_INTERVAL_SECONDS = 3
MAX_WAIT_TIME_SECONDS = 300
