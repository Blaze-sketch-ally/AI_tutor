import json

from langsmith import traceable
from models.router import route


class QueryResolverAgent:

    def __init__(
        self,
        provider="groq",
        model="llama-3.3-70b-versatile"
    ):
        self.provider = provider
        self.model = model

    @traceable(
        name="QueryResolverAgent.resolve"
    )
    def resolve(
        self,
        query: str,
        history_text: str = "",
        summary: str = ""
    ):

        prompt = f"""
You are a conversational query resolver.

Your job:

1. Determine whether the user query is a follow-up.
2. Identify the actual topic.
3. Rewrite the query into a complete standalone query.

==================================================
CONVERSATION SUMMARY
==================================================

{summary}

==================================================
CONVERSATION HISTORY
==================================================

{history_text}

==================================================
CURRENT USER QUERY
==================================================

{query}

==================================================
OUTPUT FORMAT (JSON ONLY)
==================================================

{{
    "is_followup": true,
    "resolved_topic": "...",
    "resolved_query": "..."
}}

RULES:

- If the query depends on previous context,
  set is_followup=true.

- If the query is standalone,
  set is_followup=false.

- resolved_query must always be a complete,
  retrieval-friendly query.

Examples:

Query:
advantages?

Output:
{{
    "is_followup": true,
    "resolved_topic": "CNN",
    "resolved_query": "Advantages of CNN"
}}

Query:
draw it

Output:
{{
    "is_followup": true,
    "resolved_topic": "Superheterodyne Receiver",
    "resolved_query": "Block diagram of Superheterodyne Receiver"
}}

Query:
What is Gradient Descent?

Output:
{{
    "is_followup": false,
    "resolved_topic": "Gradient Descent",
    "resolved_query": "What is Gradient Descent?"
}}

Return ONLY JSON.
"""

        try:

            response = route(
                prompt=prompt,
                provider=self.provider,
                model=self.model
            )

            return json.loads(response)

        except Exception as e:

            print(
                f"QueryResolver Error: {e}"
            )

            return {
                "is_followup": False,
                "resolved_topic": query,
                "resolved_query": query
            }