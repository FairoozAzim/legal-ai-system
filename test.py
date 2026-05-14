from app.ingestion import OCRPDFLoader

loader = OCRPDFLoader(
    "pdf_samples/sample_legal_case_packet.pdf"
)

documents = loader.load()

print(documents[0])