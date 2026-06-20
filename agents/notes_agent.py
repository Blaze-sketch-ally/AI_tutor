from langsmith import traceable
from models.router import route


class NotesAgent:

    def __init__(
        self,
        provider="groq",
        model="llama-3.3-70b-versatile"
    ):
        self.provider = provider
        self.model = model

    @traceable(name="NotesAgent.generate_notes")
    def generate_notes(
        self,
        topic: str,
        context: str = "",
        history_text: str = "",
        summary: str = "",
        strict_pdf: bool = False
    ):

        if strict_pdf:

            rule_block = """
You are an academic tutor.

STRICT RULES:

- Use ONLY PDF context.
- Do NOT use external knowledge.
- If information is missing say:
  "Not found in PDF".

- Use conversation summary and history
  to resolve references.
"""

        else:

            rule_block = """
You are Blaze AI Tutor.

RULES:

- Use PDF context if available.
- Use general knowledge if needed.
- Generate high-quality exam notes.
- Use conversation summary first.
- Use conversation history for details.
"""

        prompt = f"""
{rule_block}

==================================================
CONVERSATION SUMMARY
==================================================

{summary}

==================================================
CONVERSATION HISTORY
==================================================

{history_text}

==================================================
PDF CONTEXT
==================================================

{context}

==================================================
CURRENT REQUEST
==================================================

{topic}

==================================================
OUTPUT FORMAT
==================================================

# Topic Overview

# Key Concepts

# Important Points

# Formulas (if applicable)

# Quick Revision Notes

Generate notes now.
"""

        return route(
            prompt=prompt,
            provider=self.provider,
            model=self.model
        )