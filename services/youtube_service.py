from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

load_dotenv()


class YouTubeService:

    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")

        self.youtube = build(
            "youtube",
            "v3",
            developerKey=self.api_key
        )

    def search_videos(self, query: str, max_results: int = 5):

        request = self.youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results
        )

        response = request.execute()

        videos = []

        for item in response.get("items", []):

            video_id = item["id"]["videoId"]
            snippet = item["snippet"]

            videos.append({
                "title": snippet["title"],
                "channel": snippet["channelTitle"],
                "published_at": snippet["publishedAt"],
                "thumbnail": snippet["thumbnails"]["high"]["url"],
                "url": f"https://www.youtube.com/watch?v={video_id}"
            })

        return videos