import json

from langsmith import traceable
from models.router import route


class IntentClassifier:

    def __init__(
        self,
        provider="groq",
        model="llama-3.3-70b-versatile"
    ):
        self.provider = provider
        self.model = model

    @traceable(name="IntentClassifier.classify")
    def classify(
        self,
        query: str
    ):

        # =====================================
        # FOLLOW-UP QUESTION DETECTION
        # =====================================

        q = query.lower().strip()

        followup_patterns = [

            "explain more",
            "explain again",
            "continue",
            "elaborate",
            "expand this",
            "expand it",

            "give bigger answer",
            "bigger answer",
            "long answer",
            "detailed answer",

            "give example",
            "example",

            "why",
            "how",

            "advantages",
            "disadvantages",

            "compare",
            "compare them",

            "simplify it",
            "explain simply"
        ]

        if any(
            pattern in q
            for pattern in followup_patterns
        ):

            return {
                "chat": False,
                "answer": True,
                "quiz": False,
                "notes": False,
                "diagram": False,
                "videos": False,
                "papers": False,
                "knowledge_source": "general",
                "resource_source": "query"
            }

        # =====================================
        # LLM CLASSIFICATION
        # =====================================

        prompt = f"""
You are an intelligent intent classifier.

Your job is to classify user requests.

----------------------------------
AVAILABLE INTENTS
----------------------------------

chat
answer
quiz
notes
diagram
videos
papers

Multiple intents may be true.

----------------------------------
FOLLOW-UP QUESTION RULES
----------------------------------

If the user says:

- explain more
- explain again
- continue
- elaborate
- expand this
- give bigger answer
- give detailed answer
- long answer
- give example
- why
- how
- advantages
- disadvantages
- compare them

Then classify:

{{
    "chat": false,
    "answer": true
}}

These are educational follow-up questions.

----------------------------------
KNOWLEDGE SOURCE
----------------------------------

Choose ONE:

general
pdf
hybrid

general
- User wants normal AI knowledge.

pdf
- User specifically wants answer from uploaded PDF.

hybrid
- User wants PDF + external knowledge.

If user says:

"from pdf"
"from document"
"from uploaded notes"
"according to the pdf"

→ knowledge_source = pdf

If user says:

"using pdf and your knowledge"
"expand the pdf notes"
"compare with real world"

→ knowledge_source = hybrid

Otherwise:

→ knowledge_source = general

----------------------------------
RESOURCE SOURCE
----------------------------------

Choose ONE:

query
pdf

query:
Use query topic itself.

pdf:
Use topics extracted from uploaded PDF.

----------------------------------
EXAMPLES
----------------------------------

User: hi

{{
"chat": true,
"answer": false,
"quiz": false,
"notes": false,
"diagram": false,
"videos": false,
"papers": false,
"knowledge_source": "general",
"resource_source": "query"
}}

User: explain machine learning

{{
"chat": false,
"answer": true,
"quiz": false,
"notes": false,
"diagram": false,
"videos": false,
"papers": false,
"knowledge_source": "general",
"resource_source": "query"
}}

User: explain neural networks and give videos

{{
"chat": false,
"answer": true,
"quiz": false,
"notes": false,
"diagram": false,
"videos": true,
"papers": false,
"knowledge_source": "general",
"resource_source": "query"
}}

User: summarize uploaded pdf

{{
"chat": false,
"answer": true,
"quiz": false,
"notes": false,
"diagram": false,
"videos": false,
"papers": false,
"knowledge_source": "pdf",
"resource_source": "pdf"
}}

User: explain uploaded pdf and add examples

{{
"chat": false,
"answer": true,
"quiz": false,
"notes": false,
"diagram": false,
"videos": false,
"papers": false,
"knowledge_source": "hybrid",
"resource_source": "pdf"
}}

User: give youtube videos related to uploaded pdf

{{
"chat": false,
"answer": false,
"quiz": false,
"notes": false,
"diagram": false,
"videos": true,
"papers": false,
"knowledge_source": "pdf",
"resource_source": "pdf"
}}

----------------------------------
USER QUERY
----------------------------------

{query}

Return ONLY valid JSON.
"""

        try:

            response = route(
                prompt=prompt,
                provider=self.provider,
                model=self.model
            )

            response = response.strip()

            response = response.replace(
                "```json",
                ""
            )

            response = response.replace(
                "```",
                ""
            )

            response = response.strip()

            intents = json.loads(
                response
            )

            defaults = {
                "chat": False,
                "answer": False,
                "quiz": False,
                "notes": False,
                "diagram": False,
                "videos": False,
                "papers": False,
                "knowledge_source": "general",
                "resource_source": "query"
            }

            defaults.update(
                intents
            )

            return defaults

        except Exception as e:

            print(
                f"IntentClassifier Error: {e}"
            )

            return {
                "chat": False,
                "answer": True,
                "quiz": False,
                "notes": False,
                "diagram": False,
                "videos": False,
                "papers": False,
                "knowledge_source": "general",
                "resource_source": "query"
            }