import os
import tempfile
import pytest
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import DocumentChunker


def test_chunker_basic_split():
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
    sample_text = "This is a simple test sentence to verify that the chunker splits text properly into overlapping chunks."
    metadata = {"file_name": "test.txt", "page": 1}

    chunks = chunker.split_text(sample_text, metadata)
    assert len(chunks) > 1
    for chunk in chunks:
        assert "id" in chunk
        assert "text" in chunk
        assert chunk["metadata"]["file_name"] == "test.txt"
        assert chunk["metadata"]["page"] == 1
        assert chunk["metadata"]["word_count"] <= 10


def test_chunker_empty_text():
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.split_text("", {"file_name": "empty.txt", "page": 1})
    assert chunks == []


def test_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=10, chunk_overlap=10)


def test_parse_text_file():
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Artificial intelligence is transforming modern search pipelines.\nHybrid search combines keywords and semantics.")
        tmp_name = f.name

    try:
        parsed = DocumentParser.parse_file(tmp_name)
        assert parsed["file_type"] == "txt"
        assert parsed["total_pages"] == 1
        assert len(parsed["pages"]) == 1
        assert "Artificial intelligence" in parsed["pages"][0]["text"]
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def test_parse_csv_file():
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("name,role,department\nAlice,Engineer,AI\nBob,Manager,Data")
        tmp_name = f.name

    try:
        parsed = DocumentParser.parse_file(tmp_name)
        assert parsed["file_type"] == "csv"
        assert parsed["total_pages"] >= 1
        assert "Alice" in parsed["pages"][0]["text"]
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
