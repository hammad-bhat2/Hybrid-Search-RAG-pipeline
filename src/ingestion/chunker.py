import re
import hashlib
from typing import List, Dict, Any


class DocumentChunker:
    """
    Splits text into structured, overlapping chunks with rich metadata.
    Preserves context across chunk boundaries and provides deterministic chunk IDs.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits plain text into structured chunks with attached metadata.
        """
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        if not cleaned_text:
            return []

        words = cleaned_text.split()
        if not words:
            return []

        chunks: List[Dict[str, Any]] = []
        start = 0
        chunk_id = 0
        file_name = metadata.get("file_name", "document")
        page = metadata.get("page", 1)

        step = max(1, self.chunk_size - self.chunk_overlap)

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            # Deterministic hash ID based on filename, page, and chunk index
            unique_key = f"{file_name}_p{page}_c{chunk_id}"
            point_id = hashlib.md5(unique_key.encode("utf-8")).hexdigest()

            chunk_record = {
                "id": point_id,
                "text": chunk_text,
                "metadata": {
                    "file_name": file_name,
                    "page": page,
                    "chunk_index": chunk_id,
                    "word_count": len(chunk_words),
                    "char_count": len(chunk_text)
                }
            }
            chunks.append(chunk_record)

            if end == len(words):
                break

            chunk_id += 1
            start += step

        return chunks