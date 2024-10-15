# utils/file_loader.py
import PyPDF2
from docx import Document

def load_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def load_docx(file_path):
    text = ""
    doc = Document(file_path)
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

def load_file(file_path, file_type):
    if file_type == '.pdf':
        return load_pdf(file_path)
    elif file_type == '.docx':
        return load_docx(file_path)
    else:
        raise ValueError("Unsupported file type: {}".format(file_type))

