from typing import List, Dict, Any


class PromptBuilder:
    """
    Constructs strictly grounded prompts instructing the LLM to answer
    using only provided context chunks and to include specific source citations.
    """

    @staticmethod
    def build_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
        formatted_chunks = []
        for i, chunk in enumerate(context_chunks, start=1):
            file_name = chunk.get("file_name", "Document")
            page = chunk.get("page", 1)
            text = chunk.get("text", "").strip()
            formatted_chunks.append(
                f"[Source {i}] (Document: {file_name}, Page: {page})\n{text}"
            )

        context_str = "\n\n".join(formatted_chunks) if formatted_chunks else "No relevant context found."

        prompt = f"""You are an accurate and helpful AI assistant powered by a Hybrid Search RAG pipeline.
Your task is to answer the user's question using ONLY the retrieved context sources below.

Strict Guidelines:
1. Ground your response entirely in the provided sources. Do NOT make up information or use outside knowledge.
2. For each factual claim in your answer, cite the source number using brackets like [Source 1] or [Source 2].
3. If the answer cannot be determined from the provided context, state clearly and concisely: "I cannot find sufficient information in the provided documents to answer this question."
4. Be clear, professional, and well-structured.

Context Sources:
----------------
{context_str}
----------------

User Question: {query}

Answer:"""
        return prompt