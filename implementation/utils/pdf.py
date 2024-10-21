import pymupdf4llm
import os 

# Read PDF using pymudf4llm and return text 

def read_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    text = pymupdf4llm.to_markdown(pdf_path)
    return text