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

        for chunk in split_texts:

            chunks.append(
                {
                    "text": chunk,
                    "metadata": doc["metadata"]
                }
            )

    return chunks