# the main work of this file is indexing => mean convert file into chunks
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

pdf_path = Path(__file__).parent / "Advanced_Engineering_Mathematics__10th_Edition.pdf"

# now we need to load this file
# to load this will return pages
loader = PyPDFLoader(file_path =pdf_path)
pages = loader.load()
print(len(pages))