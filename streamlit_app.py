import os
import tempfile

import streamlit as st

from dotenv import load_dotenv

from app.ingestion import OCRPDFLoader

from app.retrieval import (
    chunk_documents,
    EmbeddingModel,
    FAISSVectorStore,
    Retriever
)

from app.draft_generation import GroundedGenerator
from app.utils.output_manager import OutputManager

# -----------------------------------
# Load environment variables
# -----------------------------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_KEY")

# -----------------------------------
# Streamlit config
# -----------------------------------

st.set_page_config(
    page_title="Legal AI System",
    layout="wide"
)

st.title("Grounded Legal AI System")

st.markdown(
    """
Upload a legal PDF and generate a grounded draft summary
with evidence-backed retrieval.
"""
)

# -----------------------------------
# Sidebar
# -----------------------------------

with st.sidebar:

    st.header("Settings")

    query = st.text_area(
        "Draft Instruction",
        value="Summarize the case."
    )

    top_k = st.slider(
        "Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=5
    )

# -----------------------------------
# File Upload
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# -----------------------------------
# Main Layout
# -----------------------------------

left_col, right_col = st.columns(2)

# -----------------------------------
# Process PDF
# -----------------------------------

if uploaded_file is None:
    st.info("Upload a PDF to begin!")
    
elif uploaded_file is not None:
    os.makedirs("pdf_samples", exist_ok=True)

    pdf_path = os.path.join(
        "pdf_samples",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:

        f.write(uploaded_file.read())

    with st.spinner("Processing PDF..."):

        # -----------------------------------
        # Ingestion
        # -----------------------------------

        loader = OCRPDFLoader(pdf_path)

        documents = loader.load()

        # -----------------------------------
        # Chunking
        # -----------------------------------

        chunks = chunk_documents(documents)

        # -----------------------------------
        # Embeddings
        # -----------------------------------

        embedding_model = EmbeddingModel()

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

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

        retriever = Retriever(
            embedding_model,
            vector_store
        )

        retrieved_chunks = retriever.retrieve(
            query,
            k=top_k
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
        output_manager = OutputManager()

        save_output = output_manager.save_output(
            pdf_name=uploaded_file.name,
            query=query,
            generated_draft=draft,
            retrieved_chunks=retrieved_chunks
        )

    # -----------------------------------
    # LEFT COLUMN
    # -----------------------------------

    with left_col:

        st.subheader("Generated Draft")

        st.markdown(draft)

    # -----------------------------------
    # RIGHT COLUMN
    # -----------------------------------

    with right_col:

        st.subheader("Retrieved Evidence")

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            with st.expander(
                f"Evidence {i} | "
                f"Page {chunk['metadata']['page']}"
            ):

                st.markdown(
                    f"""
    **Source Page:** {chunk['metadata']['page']}

    **Chunk Text:**

    {chunk['text']}
    """
    )