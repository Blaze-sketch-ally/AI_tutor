from langsmith import traceable
import arxiv

from agents.topic_extractor import TopicExtractor
from models.router import route


class PaperAgent:

    def __init__(
        self,
        max_results: int = 5
    ):
        self.max_results = max_results
        self.topic_extractor = TopicExtractor()

    def _search_arxiv(
        self,
        query: str
    ):

        papers = []

        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        client = arxiv.Client()

        for result in client.results(search):

            papers.append(
                {
                    "title": result.title,
                    "summary": result.summary,
                    "authors": [
                        author.name
                        for author in result.authors
                    ],
                    "published": str(
                        result.published
                    ),
                    "pdf_url": result.pdf_url,
                    "entry_id": result.entry_id
                }
            )

        return papers

    @traceable(name="PaperAgent.search_papers")
    def get_papers(
        self,
        query: str,
        context: str = "",
        history_text: str = "",
        summary: str = ""
    ):

        try:

            topic_prompt = f"""
Use the conversation summary and history
to determine the real topic.

==================================================
CONVERSATION SUMMARY
==================================================

{summary}

==================================================
CONVERSATION HISTORY
==================================================

{history_text}

==================================================
CURRENT REQUEST
==================================================

{query}

Return ONLY the resolved topic.
"""

            resolved_topic = route(
                prompt=topic_prompt
            ).strip()

            papers = []
            seen = set()

            if context.strip():

                try:

                    topics = (
                        self.topic_extractor.extract_topics(
                            context
                        )
                    )

                except Exception:

                    topics = [resolved_topic]

                for topic in topics:

                    results = self._search_arxiv(
                        topic
                    )

                    for paper in results:

                        key = (
                            paper["title"]
                            .lower()
                            .strip()
                        )

                        if key not in seen:

                            seen.add(key)
                            papers.append(paper)

                return papers

            return self._search_arxiv(
                resolved_topic
            )

        except Exception as e:

            print(
                f"PaperAgent Error: {e}"
            )

            return []