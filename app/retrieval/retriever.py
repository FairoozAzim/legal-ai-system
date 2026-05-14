class Retriever:

    def __init__(
        self,
        embedding_model,
        vector_store
    ):

        self.embedding_model = embedding_model

        self.vector_store = vector_store

    def retrieve(
        self,
        query,
        k=5
    ):

        query_embedding = self.embedding_model.encode(
            [query]
        )

        retrieved_chunks = self.vector_store.search(
            query_embedding,
            k=k
        )

        return retrieved_chunks