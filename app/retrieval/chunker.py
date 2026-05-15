import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100
):
    """
    Split extracted documents into smaller chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []

    for doc in documents:

        split_texts = splitter.split_text(doc["text"])
        chunk_num = len(split_texts)

        for i, chunk in enumerate(split_texts):

            chunks.append(
                {
                    "text": chunk,
                    "metadata": doc["metadata"],
                    "chunk_index": str(uuid.uuid4()),
                    "total_chunks": chunk_num  
                }
            )

    return chunks