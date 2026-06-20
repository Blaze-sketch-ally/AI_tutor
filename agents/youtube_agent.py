from langsmith import traceable

from services.youtube_service import (
    YouTubeService
)

from agents.topic_extractor import (
    TopicExtractor
)

from models.router import route


class YouTubeAgent:

    def __init__(self):

        self.youtube_service = (
            YouTubeService()
        )

        self.topic_extractor = (
            TopicExtractor()
        )

    @traceable(name="YouTubeAgent.search_videos")
    def get_videos(
        self,
        query: str,
        context: str = "",
        history_text: str = "",
        summary: str = ""
    ):

        try:

            topic_prompt = f"""
Use conversation summary and history
to determine the actual topic.

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

            videos = []
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

                    results = (
                        self.youtube_service.search_videos(
                            topic
                        )
                    )

                    for video in results:

                        key = video.get(
                            "url",
                            ""
                        )

                        if key not in seen:

                            seen.add(key)
                            videos.append(video)

                return videos

            return (
                self.youtube_service.search_videos(
                    resolved_topic
                )
            )

        except Exception as e:

            print(
                f"YouTubeAgent Error: {e}"
            )

            return []