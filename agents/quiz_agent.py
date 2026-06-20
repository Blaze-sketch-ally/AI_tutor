from langsmith import traceable
from models.router import route


class QuizAgent:

    def __init__(
        self,
        provider="groq",
        model="llama-3.3-70b-versatile"
    ):
        self.provider = provider
        self.model = model

    @traceable(name="QuizAgent.generate_quiz")
    def generate_quiz(
        self,
        topic: str,
        difficulty: str = "medium",
        context: str = "",
        history_text: str = "",
        summary: str = "",
        strict_pdf: bool = False
    ):

        if strict_pdf:

            rule_block = """
You are an exam question generator.

STRICT RULES:

- Use ONLY PDF context.
- Do NOT use outside knowledge.
- Use summary and history to resolve references.
"""

        else:

            rule_block = """
You are Blaze AI Tutor.

RULES:

- Use PDF context if available.
- Use general knowledge if required.
- Generate meaningful quizzes.
- Use summary first.
- Use history for details.
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
DIFFICULTY
==================================================

{difficulty}

==================================================
OUTPUT FORMAT
==================================================

# Multiple Choice Questions (10)

# Viva Questions (5)

# Long Answer Questions (2)

Generate the quiz now.
"""

        return route(
            prompt=prompt,
            provider=self.provider,
            model=self.model
        )