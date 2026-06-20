from langsmith import traceable
from models.router import route


class ChatAgent:

    def __init__(
        self,
        provider="groq",
        model="llama-3.3-70b-versatile"
    ):
        self.provider = provider
        self.model = model

    @traceable(
        name="ChatAgent.generate_reply"
    )
    def generate_reply(
        self,
        message: str,
        conversation_history: list = None,
        history_text: str = "",
        summary: str = ""
    ):

        if not history_text:

            history_text = ""

            if conversation_history:

                recent_history = conversation_history[-10:]

                for item in recent_history:

                    role = item.get(
                        "role",
                        "user"
                    )

                    content = item.get(
                        "content",
                        ""
                    )

                    history_text += (
                        f"{role.upper()}: "
                        f"{content}\n"
                    )

        prompt = f"""
You are Blaze AI Tutor.

PERSONALITY

- Friendly
- Helpful
- Conversational
- Patient
- Student-focused

RULES

- Use conversation summary first.
- Use conversation history for details.
- Resolve follow-up questions.
- Continue naturally.
- Never invent previous conversation details.

==================================================
CONVERSATION SUMMARY
==================================================

{summary}

==================================================
CONVERSATION HISTORY
==================================================

{history_text}

==================================================
CURRENT USER MESSAGE
==================================================

{message}

==================================================
ASSISTANT RESPONSE
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

            print(f"ChatAgent Error: {e}")

            return (
                "Sorry, I could not generate a reply."
            )