from rag.vectorstore import VectorStore


class Retriever:

    def __init__(
        self,
        session_id: str,
        top_k: int = 4
    ):

        self.vectorstore = VectorStore(
            session_id=session_id
        )

        self.top_k = top_k

    # ----------------------------------
    # RETRIEVE DOCUMENTS
    # ----------------------------------

    def retrieve(
        self,
        query: str
    ) -> list[str]:

        try:

            results = self.vectorstore.search(
                query=query,
                n_results=self.top_k
            )

            return results.get(
                "documents",
                []
            )

        except Exception as e:

            print(
                f"Retriever Error: {e}"
            )

            return []

    # ----------------------------------
    # RETRIEVE WITH SCORES
    # ----------------------------------

    def retrieve_with_scores(
        self,
        query: str
    ) -> list:

        try:

            results = self.vectorstore.search(
                query=query,
                n_results=self.top_k
            )

            documents = results.get(
                "documents",
                []
            )

            distances = results.get(
                "distances",
                []
            )

            scored_results = []

            for doc, dist in zip(
                documents,
                distances
            ):

                scored_results.append(
                    {
                        "document": doc,
                        "distance": dist
                    }
                )

            return scored_results

        except Exception as e:

            print(
                f"Retriever Error: {e}"
            )

            return []

    # ----------------------------------
    # RETRIEVE WITH METADATA
    # ----------------------------------

    def retrieve_with_metadata(
        self,
        query: str
    ) -> list:

        try:

            results = self.vectorstore.search(
                query=query,
                n_results=self.top_k
            )

            documents = results.get(
                "documents",
                []
            )

            distances = results.get(
                "distances",
                []
            )

            metadatas = results.get(
                "metadatas",
                []
            )

            output = []

            for doc, dist, meta in zip(
                documents,
                distances,
                metadatas
            ):

                output.append(
                    {
                        "document": doc,
                        "distance": dist,
                        "metadata": meta
                    }
                )

            return output

        except Exception as e:

            print(
                f"Retriever Error: {e}"
            )

            return []

    # ----------------------------------
    # GET CONTEXT
    # ----------------------------------

    def get_context(
        self,
        query: str,
        threshold: float = 1.2
    ) -> str:

        try:

            results = self.retrieve_with_scores(
                query
            )

            if not results:

                print(
                    "No retrieval results."
                )

                return ""

            relevant_chunks = []

            for item in results:

                distance = item.get(
                    "distance",
                    999
                )

                print(
                    f"Distance: {distance}"
                )

                if distance <= threshold:

                    relevant_chunks.append(
                        item["document"]
                    )

            if not relevant_chunks:

                print(
                    "No relevant PDF context found."
                )

                return ""

            print(
                f"Retrieved {len(relevant_chunks)} relevant chunks."
            )

            return "\n\n".join(
                relevant_chunks
            )

        except Exception as e:

            print(
                f"Retriever Error: {e}"
            )

            return ""

    # ----------------------------------
    # GET CONTEXT WITH SOURCES
    # ----------------------------------

    def get_context_with_sources(
        self,
        query: str,
        threshold: float = 1.2
    ):

        try:

            results = (
                self.retrieve_with_metadata(
                    query
                )
            )

            if not results:

                return {
                    "context": "",
                    "sources": []
                }

            context_chunks = []

            sources = []

            seen_sources = set()

            for item in results:

                distance = item.get(
                    "distance",
                    999
                )

                if distance > threshold:

                    continue

                context_chunks.append(
                    item["document"]
                )

                metadata = item.get(
                    "metadata",
                    {}
                )

                source_name = metadata.get(
                    "source",
                    "Unknown"
                )

                page = metadata.get(
                    "page",
                    "N/A"
                )

                key = (
                    source_name,
                    page
                )

                if key not in seen_sources:

                    seen_sources.add(
                        key
                    )

                    sources.append(
                        {
                            "source": source_name,
                            "page": page,
                            "distance": distance
                        }
                    )

            return {

                "context": "\n\n".join(
                    context_chunks
                ),

                "sources": sources
            }

        except Exception as e:

            print(
                f"Retriever Error: {e}"
            )

            return {
                "context": "",
                "sources": []
            }

    # ----------------------------------
    # PDF RELEVANCE CHECK
    # ----------------------------------

    def is_pdf_relevant(
        self,
        query: str,
        threshold: float = 1.2
    ) -> bool:

        try:

            results = self.retrieve_with_scores(
                query
            )

            if not results:

                return False

            best_distance = min(
                item["distance"]
                for item in results
            )

            print(
                f"Best PDF Distance: {best_distance}"
            )

            return best_distance <= threshold

        except Exception as e:

            print(
                f"Retriever Error: {e}"
            )

            return False

    # ----------------------------------
    # BEST DISTANCE
    # ----------------------------------

    def best_distance(
        self,
        query: str
    ):

        try:

            results = self.retrieve_with_scores(
                query
            )

            if not results:

                return None

            best = min(
                item["distance"]
                for item in results
            )

            print(
                f"Best Distance: {best}"
            )

            return best

        except Exception as e:

            print(
                f"Retriever Error: {e}"
            )

            return None

    # ----------------------------------
    # HAS DOCUMENTS
    # ----------------------------------

    def has_documents(
        self
    ) -> bool:

        try:

            count = (
                self.vectorstore.count_documents()
            )

            print(
                f"Document Count: {count}"
            )

            return count > 0

        except Exception as e:

            print(
                f"Retriever Error: {e}"
            )

            return False

    # ----------------------------------
    # DOCUMENT COUNT
    # ----------------------------------

    def count_documents(
        self
    ) -> int:

        try:

            return (
                self.vectorstore.count_documents()
            )

        except Exception as e:

            print(
                f"Retriever Error: {e}"
            )

            return 0

    # ----------------------------------
    # GET ALL DOCUMENTS
    # ----------------------------------

    def get_all_documents(
        self
    ):

        try:

            return (
                self.vectorstore.get_all_documents()
            )

        except Exception as e:

            print(
                f"Retriever Error: {e}"
            )

            return {}

    # ----------------------------------
    # CLEAR DOCUMENTS
    # ----------------------------------

    def clear_documents(
        self
    ):

        try:

            self.vectorstore.delete_all()

            print(
                "All session documents deleted."
            )

        except Exception as e:

            print(
                f"Retriever Error: {e}"
            )