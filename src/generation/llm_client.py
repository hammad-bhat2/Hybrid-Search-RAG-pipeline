import os
import re
import logging
from typing import List, Dict, Any, Optional
from config.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Multi-provider LLM client supporting OpenAI, Google Gemini, and
    an intelligent extractive fallback when external API keys are not provided.
    """

    def __init__(self):
        self.provider = self._resolve_provider()
        self._init_backend()

    def _resolve_provider(self) -> str:
        configured = settings.LLM_PROVIDER.lower()
        if configured in ["openai", "gemini", "fallback"]:
            return configured

        # Auto-detect based on available keys
        if settings.OPENAI_API_KEY and len(settings.OPENAI_API_KEY.strip()) > 5:
            return "openai"
        elif settings.GEMINI_API_KEY and len(settings.GEMINI_API_KEY.strip()) > 5:
            return "gemini"
        else:
            return "fallback"

    def _init_backend(self):
        self.openai_client = None
        self.gemini_client = None

        if self.provider == "openai":
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info(f"Initialized OpenAI client with model {settings.OPENAI_MODEL}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}. Reverting to fallback.")
                self.provider = "fallback"

        elif self.provider == "gemini":
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
                logger.info(f"Initialized Google Gemini client with model {settings.GEMINI_MODEL}")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}. Reverting to fallback.")
                self.provider = "fallback"

        if self.provider == "fallback":
            logger.info("Using grounded local extractive answer generator.")

    def generate_response(self, prompt: str, context_chunks: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generates response using active provider or fallback.
        """
        if self.provider == "openai" and self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a precise technical assistant for a hybrid search RAG pipeline."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS
                )
                return response.choices[0].message.content or "No response generated."
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}. Switching to extractive response.")
                return self._extractive_fallback(prompt, context_chunks)

        elif self.provider == "gemini" and self.gemini_client:
            try:
                response = self.gemini_client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt
                )
                return response.text or "No response generated."
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}. Switching to extractive response.")
                return self._extractive_fallback(prompt, context_chunks)

        return self._extractive_fallback(prompt, context_chunks)

    def _extractive_fallback(self, prompt: str, context_chunks: Optional[List[Dict[str, Any]]]) -> str:
        """
        Extracts and synthesizes relevant content directly from retrieved chunks.
        Used when no external LLM API key is configured or when the API fails.
        """
        if not context_chunks:
            return "I cannot find sufficient information in the provided documents to answer this question."

        # Extract question line from prompt
        match = re.search(r"User Question:\s*(.+?)(?:\nAnswer:|$)", prompt, re.DOTALL)
        query = match.group(1).strip() if match else ""
        query_words = set(re.findall(r"\w+", query.lower()))

        extracted_findings = []
        for i, chunk in enumerate(context_chunks[:3], start=1):
            text = chunk.get("text", "").strip()
            file_name = chunk.get("file_name", "Document")
            page = chunk.get("page", 1)

            # Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", text)
            scored_sentences = []
            for sent in sentences:
                sent_words = set(re.findall(r"\w+", sent.lower()))
                overlap = len(query_words.intersection(sent_words))
                if overlap > 0:
                    scored_sentences.append((overlap, sent.strip()))

            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [s[1] for s in scored_sentences[:2]]
            if not top_sentences and sentences:
                top_sentences = [sentences[0].strip()]

            if top_sentences:
                extracted_findings.append(
                    f"- {' '.join(top_sentences)} [Source {i}: {file_name}, Page {page}]"
                )

        if not extracted_findings:
            return "I cannot find sufficient information in the provided documents to answer this question."

        findings_text = "\n".join(extracted_findings)
        response = (
            f"Based on the indexed documents, here are the relevant findings for your query:\n\n"
            f"{findings_text}\n\n"
            f"> *Note: Running in offline local extraction mode. Add OPENAI_API_KEY or GEMINI_API_KEY to `.env` to enable full generative synthesis.*"
        )
        return response