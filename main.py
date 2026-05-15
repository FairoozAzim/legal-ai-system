import os
from dotenv import load_dotenv
from app.ingestion import OCRPDFLoader
from app.utils.output_manager import OutputManager
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
from test_feedack import feedback_learning

load_dotenv()  
GROQ_API_KEY = os.getenv("GROQ_KEY")
# -----------------------------------
# Load PDF
# -----------------------------------
PDF_PATH =  "./pdf_samples/PSL_Demo_Legal_Document.pdf"
PDF_NAME = Path(PDF_PATH).name

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
vector_store.save("data/vector_store/")

# -----------------------------------
# Retrieval
# -----------------------------------

query = "Summarize the document"
retriever = Retriever(
    embedding_model,
    vector_store
)

retrieved_chunks = retriever.retrieve(query,k=5)

# -----------------------------------
# Generation
# -----------------------------------

generator = GroundedGenerator(
    groq_api_key= GROQ_API_KEY
)

draft = generator.generate(
    retrieved_chunks
)

output_manager = OutputManager()
save_output = output_manager.save_output(
        pdf_name= Path(PDF_PATH).name,
        query=query,
        generated_draft=draft,    
        retrieved_chunks=retrieved_chunks
    )

print("-"*50)
print("Original Draft: \n\n",draft)
print("-"*50)

sample_draft ="""
This is a lease agreement between Landlord Westfield Realty Partners LLC and Tenant Axiom Dynamics Inc. for premises at Suite 1402, Westfield Tower, 14th floor[cite: 3, 5, 20]. The lease term commenced on January 1, 2024, and expires on December 31, 2026[cite: 7, 21]. The agreement outlines terms including a monthly base rent of $24,500, remedies for default, and governing law[cite: 7, 24, 55, 65].

Upon an event of default, the Landlord may terminate the lease, re-enter and re-let the premises, or pursue other remedies available at law or in equity[cite: 57, 58, 59]. The lease may be amended only by a written instrument signed by both parties, and any disputes shall be resolved by binding arbitration in New York County[cite: 66, 70].

Key Timeline
- November 14, 2023: Amendment #1 executed, reducing the security deposit to $36,750 [cite: 7, 29, 71]
- January 1, 2024: Commencement of the lease term [cite: 7, 21]
- December 31, 2026: Expiration Date of the initial lease term 

References
- [Evidence 1]: Article III, Sections 3.1-3.2, and 3.5-3.7 [cite: 54, 57, 65, 70]
- [Evidence 2]: Article I, Sections 1.1-1.5 [cite: 19, 21, 23, 27, 28]
- [Evidence 3]: Schedule A - Rent Schedule & Financial Summary [cite: 38, 40]
- [Evidence 4]: Schedule B - Special Conditions [cite: 42]
"""
improved_draft = feedback_learning(draft, PDF_NAME, retrieved_chunks, query="Summarize the case", sample_edit=sample_draft)

print("-"*50)
print("Improved Draft: \n\n",improved_draft)
print("-"*50)