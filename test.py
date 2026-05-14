from app.ingestion import OCRPDFLoader

from app.retrieval import (
    chunk_documents,
    EmbeddingModel,
    FAISSVectorStore,
    Retriever
)

# -----------------------------------
# Load PDF
# -----------------------------------

PDF_PATH = "./pdf_samples/sample_legal_case_packet.pdf"
loader = OCRPDFLoader(PDF_PATH)

documents = loader.load()

print("----- PDF Loaded -------")
# -----------------------------------
# Chunk documents
# -----------------------------------

chunks = chunk_documents(documents)

print(f"Total chunks: {len(chunks)}")

# -----------------------------------
# Create embeddings
# -----------------------------------

print("----- Creating Embeddings ------")
embedding_model = EmbeddingModel()

texts = [chunk["text"] for chunk in chunks]

embeddings = embedding_model.encode(texts)

# -----------------------------------
# Create vector store
# -----------------------------------

embedding_dim = embeddings.shape[1]

vector_store = FAISSVectorStore(
    embedding_dim
)

vector_store.add_documents(
    embeddings,
    chunks
)

# -----------------------------------
# Retrieval
# -----------------------------------

retriever = Retriever(
    embedding_model,
    vector_store
)

results = retriever.retrieve(
    "ownership dispute evidence",
    k=3
)

# -----------------------------------
# Print results
# -----------------------------------

for i, result in enumerate(results, start=1):

    print("=" * 80)
    print(f"RESULT {i}")
    print("=" * 80)

    print(result["text"][:1000])
    print("\n")