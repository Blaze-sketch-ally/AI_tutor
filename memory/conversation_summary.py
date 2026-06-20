from langsmith import traceable
from models.router import route


class ConversationSummaryAgent:

    def __init__(
        self,
        provider="groq",
        model="llama-3.3-70b-versatile"
    ):
        self.provider = provider
        self.model = model

    @traceable(
        name="ConversationSummaryAgent.summarize"
    )
    def summarize(
        self,
        history_text: str,
        current_query: str = ""
    ) -> str:

        # No conversation yet
        if not history_text.strip():
            return ""

        # Skip summarization for very small histories
        if len(history_text) < 300:
            return history_text

        prompt = f"""
You are a Conversation Memory Agent for an AI Tutor.

Your job is to create a concise memory summary that will be
passed to other AI Tutor agents so they can maintain context
across long conversations.

The summary should help another AI Tutor understand:

1. What topic the user is learning
2. What has already been explained
3. Important concepts already covered
4. The user's current learning objective
5. The current focus of discussion

IMPORTANT RULES:

- Keep the summary concise.
- Preserve educational context.
- Preserve user goals.
- Preserve important definitions and concepts.
- Do NOT include greetings.
- Do NOT include small talk.
- Do NOT repeat information.
- Do NOT include irrelevant details.
- Focus only on information useful for continuing the tutoring session.

==================================================
CURRENT USER QUERY
==================================================

{current_query}

==================================================
CONVERSATION HISTORY
==================================================

{history_text}

==================================================
OUTPUT FORMAT
==================================================

Current Topic:
<main topic>

Subtopics Covered:
- ...
- ...

Important Concepts:
- ...
- ...

Current User Goal:
...

Current Focus:
...

==================================================
SUMMARY
==================================================
"""

        try:

            response = route(
                prompt=prompt,
                provider=self.provider,
                model=self.model
            )

            return response.strip()

        except Exception as e:

            print(
                f"ConversationSummaryAgent Error: {e}"
            )

            return ""