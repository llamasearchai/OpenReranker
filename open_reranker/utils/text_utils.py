import re
from typing import List, Optional

import langdetect

from open_reranker.core.config import settings


def preprocess_text(text: str) -> str:
    """
    Preprocess text by cleaning up whitespace and special characters.

    Args:
        text: The input text to preprocess

    Returns:
        The preprocessed text
    """
    # Replace multiple whitespace with a single space
    text = re.sub(r"\s+", " ", text)

    # Fix Unicode escape sequences
    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)

    # Replace backslash escapes for quotes
    text = text.replace("\\'", "'").replace('\\"', '"')

    return text.strip()


def truncate_text(text: str, max_length: int) -> str:
    """
    Truncate text to the specified maximum length.

    Args:
        text: The input text to truncate
        max_length: The maximum length

    Returns:
        The truncated text
    """
    if len(text) <= max_length:
        return text

    # Try to truncate at a sentence boundary
    last_period = text.rfind(".", 0, max_length - 3)
    last_question = text.rfind("?", 0, max_length - 3)
    last_exclamation = text.rfind("!", 0, max_length - 3)

    # Find the last sentence boundary
    last_sentence_end = max(last_period, last_question, last_exclamation)

    if last_sentence_end > max_length // 2:
        # Truncate at sentence boundary and add ellipsis
        return text[: last_sentence_end + 1] + "..."
    else:
        # Truncate at max_length and add ellipsis
        return text[: max_length - 3] + "..."


def detect_language(text: str) -> Optional[str]:
    """
    Detect the language of the text.

    Args:
        text: The input text

    Returns:
        The detected language code or None if detection fails
    """
    try:
        return langdetect.detect(text)
    except:
        return None


def split_long_text(text: str, max_chunk_length: int = 512) -> List[str]:
    """
    Split long text into chunks for processing.

    Args:
        text: The input text
        max_chunk_length: Maximum length of each chunk

    Returns:
        List of text chunks
    """
    # If text is short enough, return as is
    if len(text) <= max_chunk_length:
        return [text]

    chunks = []
    current_pos = 0

    while current_pos < len(text):
        # Find a good breaking point
        if current_pos + max_chunk_length >= len(text):
            # Last chunk
            chunks.append(text[current_pos:])
            break

        # Look for sentence boundary
        end_pos = current_pos + max_chunk_length

        # Try to find the last sentence boundary
        last_period = text.rfind(".", current_pos, end_pos)
        last_question = text.rfind("?", current_pos, end_pos)
        last_exclamation = text.rfind("!", current_pos, end_pos)

        # Find the last sentence boundary
        last_sentence_end = max(last_period, last_question, last_exclamation)

        if last_sentence_end > current_pos:
            # Break at sentence boundary
            chunks.append(text[current_pos : last_sentence_end + 1])
            current_pos = last_sentence_end + 1
        else:
            # No good sentence boundary found, break at max length
            chunks.append(text[current_pos:end_pos])
            current_pos = end_pos

    return chunks
