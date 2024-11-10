import pymupdf4llm # type: ignore
import os
from datetime import datetime 
import PyPDF2
from typing import cast
from agent import MapAgent
from pipeline_types import EnrichedPdfFile

# Read PDF using pymudf4llm and return text

def read_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    text = pymupdf4llm.to_markdown(pdf_path)
    return text

def extract_title(pdf_path) -> str:
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        # Attempt to extract title from metadata
        if reader.metadata is not None:
            if "/Title" in reader.metadata and reader.metadata["/Title"]:
                return cast(str, reader.metadata["/Title"])
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using filename {filename} as title"
        )
        return filename
    
class EnrichPdfFileAgent(MapAgent[str, EnrichedPdfFile]):
    async def handle(self, filename: str) -> EnrichedPdfFile:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Enriching PDF file: {filename}"
        )
        source = extract_title(filename)
        text = read_pdf(filename)
        return EnrichedPdfFile(
            filename=filename, source=source, source_type="paper", text=text
        )
