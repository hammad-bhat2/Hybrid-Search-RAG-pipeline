import pytest
from src.generation.prompt_builder import PromptBuilder
from src.generation.llm_client import LLMClient


def test_prompt_builder_formatting():
    chunks = [
        {"file_name": "manual.pdf", "page": 3, "text": "The maximum payload capacity is 500 kilograms."},
        {"file_name": "specs.txt", "page": 1, "text": "System operates between 10C and 40C."}
    ]
    query = "What is the maximum payload capacity?"
    prompt = PromptBuilder.build_prompt(query, chunks)

    assert "[Source 1] (Document: manual.pdf, Page: 3)" in prompt
    assert "[Source 2] (Document: specs.txt, Page: 1)" in prompt
    assert "User Question: What is the maximum payload capacity?" in prompt
    assert "500 kilograms" in prompt


def test_prompt_builder_empty_context():
    prompt = PromptBuilder.build_prompt("Any question?", [])
    assert "No relevant context found." in prompt


def test_extractive_fallback_generation():
    client = LLMClient()
    # Force fallback for deterministic testing
    client.provider = "fallback"

    chunks = [
        {"file_name": "network.pdf", "page": 4, "text": "The database uses port 6333 for REST API connections."}
    ]
    prompt = PromptBuilder.build_prompt("What port does the database use?", chunks)
    response = client.generate_response(prompt=prompt, context_chunks=chunks)

    assert "6333" in response
    assert "network.pdf" in response
    assert "Page 4" in response
