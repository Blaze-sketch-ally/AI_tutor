from typing import List, Dict


class ChatMemory:

    def __init__(
        self,
        session_id: str
    ):

        self.session_id = session_id

        self.history: List[Dict] = []

    # ----------------------------------
    # ADD MESSAGE
    # ----------------------------------

    def add_message(
        self,
        role: str,
        content: str
    ):

        self.history.append(
            {
                "role": role,
                "content": content
            }
        )

    # ----------------------------------
    # GET HISTORY
    # ----------------------------------

    def get_history(
        self,
        max_turns: int = 10
    ):

        return self.history[-max_turns:]

    # ----------------------------------
    # HISTORY AS TEXT
    # ----------------------------------

    def get_history_text(
        self,
        max_turns: int = 10
    ) -> str:

        recent = self.history[-max_turns:]

        history_text = ""

        for msg in recent:

            history_text += (
                f"{msg['role'].upper()}: "
                f"{msg['content']}\n"
            )

        return history_text.strip()

    # ----------------------------------
    # LAST USER MESSAGE
    # ----------------------------------

    def get_last_user_message(self):

        for msg in reversed(self.history):

            if msg["role"] == "user":

                return msg["content"]

        return ""

    # ----------------------------------
    # TOTAL MESSAGES
    # ----------------------------------

    def size(self):

        return len(self.history)

    # ----------------------------------
    # CLEAR MEMORY
    # ----------------------------------

    def clear(self):

        self.history.clear()