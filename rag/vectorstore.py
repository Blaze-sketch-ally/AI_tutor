import chromadb
import uuid

from rag.embeddings import EmbeddingModel

class VectorStore:


    def __init__(
        self,
        session_id: str = None,
        db_path: str = "./data/chroma_db"
    ):

        self.client = chromadb.PersistentClient(
            path=db_path
        )

        # ----------------------------------
        # SESSION COLLECTION
        # ----------------------------------

        if session_id is None:

            session_id = str(
                uuid.uuid4()
            )

        self.session_id = session_id

        self.collection_name = (
            f"ai_tutor_docs_{session_id}"
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name
            )
        )

        self.embedder = EmbeddingModel()

        # DEBUG
        print("\n========== VECTORSTORE ==========")

        print(
            f"Session ID: {self.session_id}"
        )

        print(
            f"Collection: {self.collection_name}"
        )

        print(
            f"Current Documents: {self.collection.count()}"
        )

        print("=================================\n")

    # ----------------------------------
    # ADD DOCUMENTS
    # ----------------------------------

    def add_documents(
        self,
        chunks: list[str],
        ids: list[str],
        metadatas: list[dict] | None = None
    ):

        try:

            embeddings = (
                self.embedder.embed_documents(
                    chunks
                )
            )

            existing = self.collection.get()

            existing_ids = set(
                existing.get(
                    "ids",
                    []
                )
            )

            new_chunks = []
            new_ids = []
            new_embeddings = []
            new_metadatas = []

            for i, doc_id in enumerate(ids):

                if doc_id not in existing_ids:

                    new_ids.append(
                        doc_id
                    )

                    new_chunks.append(
                        chunks[i]
                    )

                    new_embeddings.append(
                        embeddings[i]
                    )

                    if metadatas:

                        new_metadatas.append(
                            metadatas[i]
                        )

            if not new_ids:

                print(
                    "No new chunks to add."
                )

                return

            self.collection.add(
                ids=new_ids,
                documents=new_chunks,
                embeddings=new_embeddings,
                metadatas=(
                    new_metadatas
                    if metadatas
                    else None
                )
            )

            print(
                f"Added {len(new_ids)} chunks."
            )

            print(
                f"Total Documents: {self.collection.count()}"
            )

        except Exception as e:

            print(
                f"Add Documents Error: {e}"
            )

    # ----------------------------------
    # SEARCH
    # ----------------------------------

    def search(
    self,
    query: str,
    n_results: int = 4,
    pdf_name: str | None = None
):

        try:

            count = self.collection.count()

            print(
                f"Searching Collection: {self.collection_name}"
            )

            print(
                f"Document Count: {count}"
            )

            if count == 0:

                return {
                    "documents": [],
                    "ids": [],
                    "distances": [],
                    "metadatas": []
                }

            query_embedding = (
                self.embedder.embed_query(
                    query
                )
            )

            query_kwargs = {
                "query_embeddings": [
                    query_embedding
                ],
                "n_results": min(
                    n_results,
                    count
                )
            }

            # -------------------------
            # PDF FILTER
            # -------------------------

            if pdf_name:

                query_kwargs["where"] = {
                    "source": pdf_name
                }
            print("\n========== FILTER DEBUG ==========")
            print("Requested PDF:", pdf_name)

            docs = self.collection.get()

            print("Stored sources:")

            for meta in docs["metadatas"][:5]:
                print(meta["source"])

            print("=================================\n")
            results = self.collection.query(
                **query_kwargs
            )

            docs = results.get(
                "documents",
                [[]]
            )[0]

            dists = results.get(
                "distances",
                [[]]
            )[0]

            metas = results.get(
                "metadatas",
                [[]]
            )[0]

            print(
                f"Retrieved {len(docs)} chunks"
            )

            print(
                f"Distances: {dists}"
            )

            return {

                "documents": docs,

                "ids": results.get(
                    "ids",
                    [[]]
                )[0],

                "distances": dists,

                "metadatas": metas
            }

        except Exception as e:

            print(
                f"Search Error: {e}"
            )

            return {
                "documents": [],
                "ids": [],
                "distances": [],
                "metadatas": []
            }

    # ----------------------------------
    # COUNT DOCUMENTS
    # ----------------------------------

    def count_documents(self):

        return self.collection.count()

    # ----------------------------------
    # DELETE ALL DOCUMENTS
    # ----------------------------------

    def delete_all(self):

        try:

            data = self.collection.get()

            ids = data.get(
                "ids",
                []
            )

            if ids:

                self.collection.delete(
                    ids=ids
                )

                print(
                    f"Deleted {len(ids)} documents."
                )

        except Exception as e:

            print(
                f"Delete Error: {e}"
            )

    # ----------------------------------
    # GET ALL DOCUMENTS
    # ----------------------------------

    def get_all_documents(self):

        try:

            return self.collection.get()

        except Exception as e:

            print(
                f"Get Documents Error: {e}"
            )

            return {}

    # ----------------------------------
    # DELETE COLLECTION
    # ----------------------------------

    def delete_collection(self):

        try:

            self.client.delete_collection(
                self.collection_name
            )

            print(
                f"Deleted Collection: {self.collection_name}"
            )

        except Exception as e:

            print(
                f"Collection Delete Error: {e}"
            )

    # ----------------------------------
    # LIST COLLECTIONS
    # ----------------------------------

    def list_collections(self):

        try:

            return (
                self.client.list_collections()
            )

        except Exception as e:

            print(
                f"List Collections Error: {e}"
            )

            return []

