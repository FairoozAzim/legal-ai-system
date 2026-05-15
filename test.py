import os
from dotenv import load_dotenv
from app.ingestion import OCRPDFLoader
from app.retrieval import (
    chunk_documents,
    EmbeddingModel,
    FAISSVectorStore,
    Retriever
)

from app.feedback import (
    FeedbackStore,
    FeedbackLearner
)
from pathlib import Path

from app.draft_generation import GroundedGenerator

load_dotenv()  
GROQ_API_KEY = os.getenv("GROQ_KEY")
# -----------------------------------
# Load PDF
# -----------------------------------
PDF_PATH =  "./pdf_samples/sample_legal_case_packet.pdf"
loader = OCRPDFLoader(PDF_PATH)

documents = loader.load()

# -----------------------------------
# Chunking
# -----------------------------------

chunks = chunk_documents(documents)

# -----------------------------------
# Embeddings
# -----------------------------------

embedding_model = EmbeddingModel()

texts = [chunk["text"] for chunk in chunks]

embeddings = embedding_model.encode(texts)

# -----------------------------------
# Vector Store
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

retrieved_chunks = retriever.retrieve(
    "Summarize the ownership dispute",
    k=5
)

# -----------------------------------
# Generation
# -----------------------------------

generator = GroundedGenerator(
    groq_api_key= GROQ_API_KEY
)

draft = generator.generate(
    retrieved_chunks
)



# -----------------------------------
# Store feedback
# -----------------------------------
EDITS_FILE = Path("edit_records.json")
RULES_FILE = Path("learned_rules.json")


feedback_store = FeedbackStore()

feedback_store.capture_edit(
    original=draft,
    edited="""
        The ownership status remains uncertain.
        According to the transfer record,
        verification is still under review.
""",
    doc_id="sample_case_001",
    query="Summarize ownership dispute"
)

# -----------------------------------
# Learn rules
# -----------------------------------

learner = FeedbackLearner(
    api_key=GROQ_API_KEY,
    feedback_store=feedback_store
)

learner.process_pending_edits()

# -----------------------------------
# Build prompt injection block
# -----------------------------------

rules_block = learner.build_rules_block()

print(rules_block)


# -----------------------------------
# Generation
# -----------------------------------


generator = GroundedGenerator(
    groq_api_key= GROQ_API_KEY
)

improved_draft = generator.generate(
    retrieved_chunks,
    rules_block=rules_block
)


print("Original Draft : ", draft)
print("Improved Draft : ", improved_draft)