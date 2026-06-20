import re

from langsmith import traceable
from models.router import route


class TutorAgent:

    def __init__(
        self,
        provider="groq",
        model="llama-3.3-70b-versatile"
    ):
        self.provider = provider
        self.model = model

    # =====================================
    # AUTO MARKS DETECTION
    # =====================================

    def detect_marks(
        self,
        question: str
    ) -> str:

        q = question.lower()

        match = re.search(
            r"(3|5|6|8|10|15)\s*mark",
            q
        )

        if match:
            return match.group(1)

        if "short note" in q:
            return "5"

        if "brief" in q:
            return "3"

        if "very short" in q:
            return "3"

        return "8"

    # =====================================
    # ANSWER GENERATION
    # =====================================

    @traceable(
        name="TutorAgent.generate_answer"
    )
    def generate_answer(
        self,
        question: str,
        context: str = "",
        history_text: str = "",
        summary: str = "",
        marks: str = None,
        strict_pdf: bool = False
    ):

        # =====================================
        # MARKS DETECTION
        # =====================================

        if marks is None:

            marks = self.detect_marks(
                question
            )

        # =====================================
        # KNOWLEDGE MODE
        # =====================================

        if strict_pdf:

            if not context.strip():

                return (
                    "I could not find relevant information "
                    "inside the uploaded PDF."
                )

            rule_block = """
You are Blaze AI Tutor.

Answer ONLY from the provided PDF context.

STRICT RULES:

- Use ONLY PDF information.
- Do NOT use external knowledge.
- Do NOT invent facts.
- If information is missing say:

"This information is not available in the uploaded PDF."

- Maintain a conversational tutoring style.
- Use conversation summary and history to resolve references.
"""

        elif context.strip():

            rule_block = """
You are Blaze AI Tutor.

Use PDF context as the PRIMARY source.

You may supplement with your own knowledge
when useful.

RULES:

- Combine PDF information and general knowledge.
- Explain concepts clearly.
- Use examples whenever useful.
- Maintain a conversational tutoring style.
- Use conversation summary and history when relevant.
"""

        else:

            rule_block = """
You are Blaze AI Tutor.

Use your own knowledge.

RULES:

- Be educational.
- Be conversational.
- Explain concepts clearly.
- Use examples.
- Use conversation summary and history when relevant.
"""

        # =====================================
        # MARKS INSTRUCTION
        # =====================================

        marks_instruction = f"""
Generate an answer suitable for {marks} marks.

3 Marks:
- concise
- 3-4 points

5 Marks:
- short explanation
- key concepts

6 Marks:
- moderate detail

8 Marks:
- detailed explanation
- examples

10 Marks:
- comprehensive explanation
- examples
- applications

15 Marks:
- full descriptive answer
- advantages
- disadvantages
- applications
- conclusion
"""

        # =====================================
        # FINAL PROMPT
        # =====================================

        prompt = f"""
{rule_block}

{marks_instruction}

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
FOLLOW-UP RESOLUTION RULES
==================================================

If the question is a follow-up such as:

- why?
- how?
- explain more
- continue
- give example
- advantages?
- disadvantages?
- compare them
- diagram of it
- applications?

Then use the conversation summary and history
to determine the actual topic.

Do NOT ask the user to repeat context if it
already exists.

==================================================
QUESTION
==================================================

{question}

==================================================
ANSWER
==================================================

Provide a clear, structured educational answer.
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
                f"TutorAgent Error: {e}"
            )

            return (
                "Sorry, I could not generate an answer."
            )