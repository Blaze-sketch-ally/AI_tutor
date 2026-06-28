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

    @traceable(
        name="IntentClassifier.classify"
    )
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
                "comparison": False,
                "all_pdfs": False,
                "knowledge_source": "general",
                "resource_source": "query"
            }
        comparison_keywords = [
        "compare",
        "comparison",
        "difference between",
        "differences",
        "similarities"
    ]

        if any(k in q for k in comparison_keywords):

            is_all_pdf = (

            "all pdf" in q
            or "all uploaded pdf" in q
            or "every pdf" in q
            or "all documents" in q
            or"cross pdf"in q
            or "combined notes"in q
            or "combined quiz"in q

    )

            return {
                "chat": False,
                "answer": False,
                "quiz": False,
                "notes": False,
                "diagram": False,
                "videos": False,
                "papers": False,
                "comparison": True,
                "all_pdfs": is_all_pdf,
                "knowledge_source": (
                    "pdf"
                    if is_all_pdf
                    else "general"
                ),
                "resource_source": (
                    "pdf"
                    if is_all_pdf
                    else "query"
                )
            }
        # =====================================
        # CROSS PDF NOTES
        # =====================================

        if (

            ("notes" in q and "all pdf" in q)

            or ("notes" in q and "every pdf" in q)

            or ("notes" in q and "all uploaded pdf" in q)

            or ("combined notes" in q)

            or ("cross pdf notes" in q)

            or ("generate combined notes" in q)

            or ("create combined notes" in q)

            or ("notes from all pdfs" in q)

            or ("notes from every pdf" in q)

        ):

            return {
                "chat": False,
                "answer": False,
                "quiz": False,
                "notes": True,
                "diagram": False,
                "videos": False,
                "papers": False,
                "comparison": False,
                "all_pdfs": True,
                "knowledge_source": "pdf",
                "resource_source": "pdf"
            }


        # =====================================
        # CROSS PDF QUIZ
        # =====================================

        if (

            ("quiz" in q and "all pdf" in q)

            or ("quiz" in q and "every pdf" in q)

            or ("quiz" in q and "all uploaded pdf" in q)

            or ("combined quiz" in q)

            or ("cross pdf quiz" in q)

            or ("generate combined quiz" in q)

            or ("create combined quiz" in q)

            or ("quiz from all pdfs" in q)

            or ("quiz from every pdf" in q)

        ):

            return {
                "chat": False,
                "answer": False,
                "quiz": True,
                "notes": False,
                "diagram": False,
                "videos": False,
                "papers": False,
                "comparison": False,
                "all_pdfs": True,
                "knowledge_source": "pdf",
                "resource_source": "pdf"
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
comparison

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
- simplify it
- explain simply

Then classify:

{{
    "chat": false,
    "answer": true
}}

These are educational follow-up questions.

----------------------------------
COMPARE INTENT
----------------------------------

If the user wants:

- comparison
- difference between
- similarities and differences
- compare concepts
- compare pdfs
- compare topics

Then:

{{
    "comparison": true
}}

Examples:

Compare CNN and RNN

Compare CNN from uploaded PDFs

Difference between supervised and unsupervised learning

Compare PDF A and PDF B

----------------------------------
ALL PDFS INTENT
----------------------------------

If the user says:

- all uploaded PDFs
- all PDFs
- every PDF
- from all documents

Notes-related:
- combined notes
- cross pdf notes
- notes from all pdfs
- notes from every pdf

Quiz-related:
- combined quiz
- cross pdf quiz
- quiz from all pdfs
- quiz from every pdf

Then:

{{
    "all_pdfs": true
}}
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
"comparison": false,
"all_pdfs": false,
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
"comparison": false,
"all_pdfs": false,
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
"comparison": false,
"all_pdfs": false,
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
"comparison": false,
"all_pdfs": false,
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
"comparison": false,
"all_pdfs": false,
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
"comparison": false,
"all_pdfs": false,
"knowledge_source": "pdf",
"resource_source": "pdf"
}}

User: Generate notes from all uploaded PDFs

{{
"chat": false,
"answer": false,
"quiz": false,
"notes": true,
"diagram": false,
"videos": false,
"papers": false,
"comparison": false,
"all_pdfs": true,
"knowledge_source": "pdf",
"resource_source": "pdf"
}}

User: Generate quiz from all PDFs

{{
"chat": false,
"answer": false,
"quiz": true,
"notes": false,
"diagram": false,
"videos": false,
"papers": false,
"comparison": false,
"all_pdfs": true,
"knowledge_source": "pdf",
"resource_source": "pdf"
}}
User: Create combined notes from all uploaded PDFs

{{
"chat": false,
"answer": false,
"quiz": false,
"notes": true,
"diagram": false,
"videos": false,
"papers": false,
"comparison": false,
"all_pdfs": true,
"knowledge_source": "pdf",
"resource_source": "pdf"
}}
User: Generate quiz from all uploaded PDFs

{{
"chat": false,
"answer": false,
"quiz": true,
"notes": false,
"diagram": false,
"videos": false,
"papers": false,
"comparison": false,
"all_pdfs": true,
"knowledge_source": "pdf",
"resource_source": "pdf"
}}
User: Compare adaptive modulation from all uploaded PDFs

{{
"chat": false,
"answer": false,
"quiz": false,
"notes": false,
"diagram": false,
"videos": false,
"papers": false,
"comparison": true,
"all_pdfs": true,
"knowledge_source": "pdf",
"resource_source": "pdf"
}}

User: Compare CNN and RNN

{{
"chat": false,
"answer": false,
"quiz": false,
"notes": false,
"diagram": false,
"videos": false,
"papers": false,
"comparison": true,
"all_pdfs": false,
"knowledge_source": "general",
"resource_source": "query"
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
                "comparison": False,
                "all_pdfs": False,
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
                "comparison": False,
                "all_pdfs": False,
                "knowledge_source": "general",
                "resource_source": "query"
            }