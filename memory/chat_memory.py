from typing import List, Dict, Optional

from memory.session_manager import SessionManager


class ChatMemory:

    def __init__(
        self,
        username: str,
        session_id: str
    ):

        self.username = username
        self.session_id = session_id

        self.session_manager = SessionManager(
            username=username
        )

        # ----------------------------------
        # LOAD EXISTING HISTORY
        # ----------------------------------

        self.history: List[Dict] = (
            self.session_manager.get_history(
                session_id
            )
        )

    # ----------------------------------
    # SAVE MEMORY
    # ----------------------------------

    def save(self):

        self.session_manager.save_history(
            session_id=self.session_id,
            history=self.history
        )

    # ----------------------------------
    # LOAD MEMORY
    # ----------------------------------

    def load(self):

        self.history = (
            self.session_manager.get_history(
                self.session_id
            )
        )

    # ----------------------------------
    # ADD MESSAGE (Backward Compatible)
    # ----------------------------------

    def add_message(
        self,
        role: Optional[str] = None,
        content: Optional[str] = None,
        message: Optional[Dict] = None
    ):

        # New structured message
        if message is not None:

            self.history.append(message)

        # Old style
        else:

            self.history.append(
                {
                    "role": role,
                    "content": content
                }
            )

        self.save()

    # ----------------------------------
    # UPDATE LAST MESSAGE
    # ----------------------------------

    def update_last_message(
        self,
        updates: Dict
    ):

        if not self.history:

            return

        self.history[-1].update(
            updates
        )

        self.save()

    # ----------------------------------
    # GET HISTORY
    # ----------------------------------

    def get_history(
        self,
        max_turns: int = 10
    ):

        return self.history[-max_turns:]

    # ----------------------------------
    # GET FULL HISTORY
    # ----------------------------------

    def get_full_history(self):

        return self.history

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

            role = msg.get(
                "role",
                "assistant"
            )

            content = msg.get(
                "content",
                ""
            )

            history_text += (
                f"{role.upper()}: "
                f"{content}\n"
            )

        return history_text.strip()

    # ----------------------------------
    # LAST USER MESSAGE
    # ----------------------------------

    def get_last_user_message(self):

        for msg in reversed(
            self.history
        ):

            if msg.get(
                "role"
            ) == "user":

                return msg.get(
                    "content",
                    ""
                )

        return ""

    # ----------------------------------
    # LAST ASSISTANT MESSAGE
    # ----------------------------------

    def get_last_assistant_message(self):

        for msg in reversed(
            self.history
        ):

            if msg.get(
                "role"
            ) == "assistant":

                return msg.get(
                    "content",
                    ""
                )

        return ""

    # ----------------------------------
    # LAST ASSISTANT OBJECT
    # ----------------------------------

    def get_last_assistant_message_object(self):

        for msg in reversed(
            self.history
        ):

            if msg.get(
                "role"
            ) == "assistant":

                return msg

        return {}

    # ----------------------------------
    # TOTAL MESSAGES
    # ----------------------------------

    def size(self):

        return len(
            self.history
        )

    # ----------------------------------
    # CLEAR MEMORY
    # ----------------------------------

    def clear(self):

        self.history = []

        self.save()