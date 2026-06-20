from langsmith import traceable
from models.router import route


class DiagramAgent:

    def __init__(
        self,
        provider="groq",
        model="llama-3.3-70b-versatile"
    ):
        self.provider = provider
        self.model = model

    @traceable(
        name="DiagramAgent.generate_diagram"
    )
    def generate_diagram(
        self,
        topic: str,
        context: str = "",
        history_text: str = "",
        summary: str = "",
        strict_pdf: bool = False
    ):

        if strict_pdf:

            rule_block = """
You are an expert diagram generator.

RULES:
- Use ONLY PDF context.
- Use summary and history to resolve references.
- Do NOT use external knowledge.
"""

        else:

            rule_block = """
You are Blaze AI Tutor.

RULES:
- Use PDF context if available.
- Use general knowledge if required.
- Use summary and history to resolve references.
- Generate educational diagrams.
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
OUTPUT RULES
==================================================

- Output ONLY Mermaid code
- Use flowchart TD
- No markdown
- No explanations
- No comments

Generate the diagram now.
"""

        return route(
            prompt=prompt,
            provider=self.provider,
            model=self.model
        )