import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from google import genai
from google.genai import types

from config import POLLING_INTERVAL_SECONDS, MAX_WAIT_TIME_SECONDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cosine_similarity(a: Any, b: Any) -> float:
    """
    Compute cosine similarity between two vectors.
    Handles various input types and edge cases robustly.
    """
    try:
        a = np.asarray(a, dtype=np.float64).flatten()
        b = np.asarray(b, dtype=np.float64).flatten()
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting vectors to numpy arrays: {e}")
        return 0.0

    if a.shape != b.shape:
        logger.error(f"Vector shapes mismatch: {a.shape} vs {b.shape}")
        return 0.0

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        logger.warning("One or both vectors have zero norm")
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def wait_for_file_processing(
    client: genai.Client,
    file_name: str,
    progress_bar: Optional[Any] = None,
    status_text: Optional[Any] = None
) -> bool:
    """
    Wait for a Gemini file to finish processing.
    Returns True if processed successfully, False if failed or timed out.
    """
    elapsed_time = 0
    while elapsed_time < MAX_WAIT_TIME_SECONDS:
        try:
            g_file = client.files.get(name=file_name)
            state = getattr(g_file, "state", None)
            
            if state is None:
                logger.warning(f"File {file_name} has no state attribute, assuming ready")
                return True

            state_str = str(state)
            
            if "ACTIVE" in state_str or "READY" in state_str:
                logger.info(f"File {file_name} is ready")
                return True
            
            if "FAILED" in state_str or "ERROR" in state_str:
                logger.error(f"File {file_name} processing failed: {state}")
                return False

            if progress_bar and status_text:
                progress_bar.progress(
                    min(elapsed_time / MAX_WAIT_TIME_SECONDS, 1.0),
                    text=f"Processing {file_name}... ({elapsed_time}s elapsed)"
                )

            time.sleep(POLLING_INTERVAL_SECONDS)
            elapsed_time += POLLING_INTERVAL_SECONDS

        except Exception as e:
            logger.error(f"Error checking file state: {e}")
            time.sleep(POLLING_INTERVAL_SECONDS)
            elapsed_time += POLLING_INTERVAL_SECONDS

    logger.error(f"Timeout waiting for file {file_name} to process")
    return False


def upload_and_embed_file(
    client: genai.Client,
    file_path: str,
    display_name: str,
    model_name: str,
    progress_bar: Optional[Any] = None,
    status_text: Optional[Any] = None
) -> Tuple[Optional[List[float]], Optional[str]]:
    """
    Upload a file to Gemini, wait for processing, generate embedding, and cleanup.
    Returns (embedding_vector, file_uri) on success, (None, None) on failure.
    """
    g_file = None
    try:
        if progress_bar and status_text:
            status_text.text(f"Uploading {display_name}...")
        
        g_file = client.files.upload(
            file=file_path,
            config=types.UploadFileConfig(display_name=display_name)
        )
        
        logger.info(f"Uploaded {display_name}, file name: {g_file.name}")

        if not wait_for_file_processing(client, g_file.name, progress_bar, status_text):
            return None, None

        if progress_bar and status_text:
            status_text.text(f"Embedding {display_name}...")

        result = client.models.embed_content(
            model=model_name,
            contents=g_file
        )

        if not result.embeddings or len(result.embeddings) == 0:
            logger.error(f"No embeddings returned for {display_name}")
            return None, None

        embedding_vector = result.embeddings[0].values
        logger.info(f"Successfully embedded {display_name}")

        return embedding_vector, g_file.uri

    except Exception as e:
        logger.error(f"Error processing {display_name}: {e}")
        return None, None
    
    finally:
        if g_file:
            try:
                client.files.delete(name=g_file.name)
                logger.info(f"Deleted Gemini file: {g_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete Gemini file {g_file.name}: {e}")


def compute_similarities(
    query_embedding: List[float],
    library_items: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Compute cosine similarities between query and all library items.
    Returns sorted list of (similarity, item) tuples.
    """
    similarities = []
    for item in library_items:
        sim = cosine_similarity(query_embedding, item["embedding"])
        similarities.append((sim, item))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]


def get_file_type_category(mime_type: str) -> str:
    """Get a human-readable category for the file type."""
    if mime_type.startswith("image/"):
        return "Image"
    elif mime_type.startswith("video/"):
        return "Video"
    elif mime_type.startswith("audio/"):
        return "Audio"
    elif mime_type == "application/pdf":
        return "PDF"
    elif mime_type == "text/plain":
        return "Text"
    else:
        return "File"
