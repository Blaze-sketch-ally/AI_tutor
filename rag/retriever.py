from rag.vectorstore import VectorStore


class Retriever:

    def __init__(
        self,
        session_id: str,
        top_k: int = 5
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
    query: str,
    selected_pdfs: list[str] | None = None,
    top_k: int | None = None
):

        try:

            # --------------------------
            # NORMAL RETRIEVAL
            # --------------------------

            if not selected_pdfs:

                results = self.vectorstore.search(
                    query=query,
                    n_results=top_k or self.top_k
                )

                return results.get(
                    "documents",
                    []
                )

            # --------------------------
            # BALANCED RETRIEVAL
            # --------------------------

            total_k = top_k or self.top_k

            per_pdf_k = max(
                1,
                total_k // len(selected_pdfs)
            )

            all_documents = []

            for pdf_name in selected_pdfs:

                results = self.vectorstore.search(
                    query=query,
                    pdf_name=pdf_name,
                    n_results=per_pdf_k
                )

                docs = results.get(
                    "documents",
                    []
                )

                all_documents.extend(docs)

            return all_documents

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
    query: str,
    selected_pdfs: list[str] | None = None,
    top_k: int | None = None
):

        try:

            if not selected_pdfs:

                results = self.vectorstore.search(
                    query=query,
                    n_results=top_k or self.top_k
                )

                documents = results.get(
                    "documents",
                    []
                )

                distances = results.get(
                    "distances",
                    []
                )

            else:

                documents = []
                distances = []

                total_k = top_k or self.top_k

                per_pdf_k = max(
                    1,
                    total_k // len(selected_pdfs)
                )

                for pdf_name in selected_pdfs:

                    pdf_results = self.vectorstore.search(
                        query=query,
                        pdf_name=pdf_name,
                        n_results=per_pdf_k
                    )

                    documents.extend(
                        pdf_results.get(
                            "documents",
                            []
                        )
                    )

                    distances.extend(
                        pdf_results.get(
                            "distances",
                            []
                        )
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

            scored_results.sort(
                key=lambda x: x["distance"]
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
    query: str,
    selected_pdfs: list[str] | None = None,
    top_k: int | None = None
):

        try:

            if not selected_pdfs:

                results = self.vectorstore.search(
                    query=query,
                    n_results=top_k or self.top_k
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

            else:

                documents = []
                distances = []
                metadatas = []

                total_k = top_k or self.top_k

                per_pdf_k = max(
                    1,
                    total_k // len(selected_pdfs)
                )

                for pdf_name in selected_pdfs:

                    pdf_results = self.vectorstore.search(
                        query=query,
                        pdf_name=pdf_name,
                        n_results=per_pdf_k
                    )

                    documents.extend(
                        pdf_results.get(
                            "documents",
                            []
                        )
                    )

                    distances.extend(
                        pdf_results.get(
                            "distances",
                            []
                        )
                    )

                    metadatas.extend(
                        pdf_results.get(
                            "metadatas",
                            []
                        )
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
                        "metadata": meta or {}
                    }
                )

            output.sort(
                key=lambda x: x["distance"]
            )

            return output

        except Exception as e:

            print(
                f"Retriever Error: {e}"
            )

            return []

    # ----------------------------------
    # GET CONTEXT ONLY
    # ----------------------------------

    def get_context(
        self,
        query: str,
        threshold: float = 3,
        selected_pdfs:list[str]|None=None,
        top_k:int|None=None
        
    ) -> str:

        try:

            results = self.retrieve_with_scores(
                query=query,
                selected_pdfs= selected_pdfs,
                top_k=top_k
                
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
    # GET CONTEXT + SOURCES
    # ----------------------------------

    def get_context_with_sources(
        self,
        query: str,
        threshold: float = 3,
        top_k:int|None=None,
        selected_pdfs:list[str]|None=None
    ):

        try:

            results = (
                self.retrieve_with_metadata(
                    query=query,
                    top_k=top_k,
                    selected_pdfs=selected_pdfs
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

                chunk_id = metadata.get(
                    "chunk_id",
                    "N/A"
                )

                key = (
                    source_name,
                    page,
                    chunk_id
                )

                if key not in seen_sources:

                    seen_sources.add(
                        key
                    )

                    sources.append(
                        {
                            "source": source_name,
                            "page": page,
                            "chunk_id": chunk_id,
                            "distance": round(
                                distance,
                                4
                            )
                        }
                    )

            sources = sorted(
                sources,
                key=lambda x: x["distance"]
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
    # GROUP CONTEXT BY PDF
    # ----------------------------------

    def get_grouped_context_by_source(
        self,
        query: str,
        threshold: float = 3,
        top_k: int|None=None,
        selected_pdfs:list[str]|None=None
    ):

        try:

            results = self.retrieve_with_metadata(
                query=query,
                top_k= top_k,
                selected_pdfs=selected_pdfs
            )

            if not results:

                return {
                    "pdf_contexts": {},
                    "sources": []
                }

            pdf_contexts = {}

            sources = []

            for item in results:

                distance = item.get(
                    "distance",
                    999
                )

                if distance > threshold:

                    continue

                metadata = item.get(
                    "metadata",
                    {}
                )

                source_name = metadata.get(
                    "source",
                    "Unknown PDF"
                )

                page = metadata.get(
                    "page",
                    "N/A"
                )

                chunk_id = metadata.get(
                    "chunk_id",
                    "N/A"
                )

                document = item.get(
                    "document",
                    ""
                )

                if source_name not in pdf_contexts:

                    pdf_contexts[source_name] = []

                pdf_contexts[source_name].append(
                    document
                )

                sources.append(
                    {
                        "source": source_name,
                        "page": page,
                        "chunk_id": chunk_id,
                        "distance": round(
                            distance,
                            4
                        )
                    }
                )

            final_contexts = {}

            for pdf_name, chunks in (
                pdf_contexts.items()
            ):

                final_contexts[pdf_name] = (
                    "\n\n".join(chunks)
                )

            return {

                "pdf_contexts": final_contexts,

                "sources": sorted(
                    sources,
                    key=lambda x: x["distance"]
                )
            }

        except Exception as e:

            print(
                f"Retriever Error: {e}"
            )

            return {
                "pdf_contexts": {},
                "sources": []
            }

    # ----------------------------------
    # PDF RELEVANCE CHECK
    # ----------------------------------

    def is_pdf_relevant(
        self,
        query: str,
        threshold: float = 3,
        selected_pdfs:list[str]|None=None
    ) -> bool:

        try:

            results = self.retrieve_with_scores(
                query=query,
                selected_pdfs=selected_pdfs
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