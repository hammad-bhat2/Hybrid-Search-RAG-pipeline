import os
from typing import Dict, List, Any
from pathlib import Path


class DocumentParser:
    """
    Multi-format document parser supporting PDF, DOCX, TXT, Markdown, and CSV.
    Extracts text organized by page or logical section with metadata.
    """

    @classmethod
    def parse_file(cls, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        file_name = path.name

        if ext == ".pdf":
            return cls._parse_pdf(path)
        elif ext in [".docx", ".doc"]:
            return cls._parse_docx(path)
        elif ext in [".txt", ".md", ".markdown"]:
            return cls._parse_text(path)
        elif ext == ".csv":
            return cls._parse_csv(path)
        else:
            # Fallback to plain text reading
            return cls._parse_text(path)

    @staticmethod
    def _parse_pdf(path: Path) -> Dict[str, Any]:
        pages_data: List[Dict[str, Any]] = []
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            for idx, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages_data.append({
                        "page": idx,
                        "text": text
                    })
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF '{path.name}': {str(e)}") from e

        if not pages_data:
            pages_data.append({"page": 1, "text": ""})

        return {
            "file_name": path.name,
            "file_type": "pdf",
            "total_pages": len(pages_data),
            "pages": pages_data
        }

    @staticmethod
    def _parse_docx(path: Path) -> Dict[str, Any]:
        pages_data: List[Dict[str, Any]] = []
        try:
            import docx
            doc = docx.Document(str(path))
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())

            # Group paragraphs into logical pages (approx. 400 words per page)
            words_per_page = 400
            current_page_text: List[str] = []
            current_word_count = 0
            page_num = 1

            for para in full_text:
                para_words = len(para.split())
                current_page_text.append(para)
                current_word_count += para_words

                if current_word_count >= words_per_page:
                    pages_data.append({
                        "page": page_num,
                        "text": "\n\n".join(current_page_text)
                    })
                    page_num += 1
                    current_page_text = []
                    current_word_count = 0

            if current_page_text:
                pages_data.append({
                    "page": page_num,
                    "text": "\n\n".join(current_page_text)
                })
        except Exception as e:
            raise RuntimeError(f"Failed to parse DOCX '{path.name}': {str(e)}") from e

        if not pages_data:
            pages_data.append({"page": 1, "text": ""})

        return {
            "file_name": path.name,
            "file_type": "docx",
            "total_pages": len(pages_data),
            "pages": pages_data
        }

    @staticmethod
    def _parse_text(path: Path) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()
        except Exception as e:
            raise RuntimeError(f"Failed to read text file '{path.name}': {str(e)}") from e

        # If long, divide into ~500 word logical pages
        words = content.split()
        words_per_page = 500
        pages_data = []

        if not words:
            pages_data.append({"page": 1, "text": ""})
        else:
            for i in range(0, len(words), words_per_page):
                page_text = " ".join(words[i:i + words_per_page])
                pages_data.append({
                    "page": (i // words_per_page) + 1,
                    "text": page_text
                })

        return {
            "file_name": path.name,
            "file_type": path.suffix.replace(".", "").lower() or "txt",
            "total_pages": len(pages_data),
            "pages": pages_data
        }

    @staticmethod
    def _parse_csv(path: Path) -> Dict[str, Any]:
        import csv
        rows = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                header_str = ", ".join(header) if header else ""
                for row_idx, row in enumerate(reader, start=1):
                    row_str = f"Row {row_idx}: " + ", ".join([f"{h}={v}" for h, v in zip(header or [], row)])
                    rows.append(row_str)
        except Exception as e:
            raise RuntimeError(f"Failed to parse CSV '{path.name}': {str(e)}") from e

        # Group rows into pages of ~20 rows
        rows_per_page = 20
        pages_data = []
        for i in range(0, max(1, len(rows)), rows_per_page):
            chunk_rows = rows[i:i + rows_per_page]
            pages_data.append({
                "page": (i // rows_per_page) + 1,
                "text": "\n".join(chunk_rows)
            })

        return {
            "file_name": path.name,
            "file_type": "csv",
            "total_pages": len(pages_data),
            "pages": pages_data
        }
