import os
from datetime import datetime

from database.supabase_client import supabase


class SessionManager:

    def __init__(
        self,
        username: str
    ):

        self.username = username.strip().lower()

        if not self.username:

            raise ValueError(
                "Username cannot be empty."
            )

        # ----------------------------------
        # FETCH USER ID
        # ----------------------------------

        result = (
            supabase
            .table("users")
            .select("id")
            .eq(
                "username",
                self.username
            )
            .execute()
        )

        if not result.data:

            raise ValueError(
                f"User '{self.username}' not found."
            )

        self.user_id = result.data[0]["id"]

    # =====================================
    # CREATE SESSION
    # =====================================

    def create_session(
        self,
        title: str = "New Chat"
    ):

        result = (
            supabase
            .table("sessions")
            .insert(
                {
                    "user_id": self.user_id,
                    "title": title,
                    "summary": ""
                }
            )
            .execute()
        )

        return result.data[0]["id"]

    # =====================================
    # LOAD SESSION
    # =====================================

    def load_session(
        self,
        session_id: str
    ):

        result = (
            supabase
            .table("sessions")
            .select("*")
            .eq(
                "id",
                session_id
            )
            .eq(
                "user_id",
                self.user_id
            )
            .execute()
        )

        if not result.data:

            return None

        session = result.data[0]

        return {

            "session_id": session["id"],

            "title": session["title"],

            "summary": session.get(
                "summary",
                ""
            ),

            "created_at": session[
                "created_at"
            ]
        }

    # =====================================
    # SESSION EXISTS
    # =====================================

    def session_exists(
        self,
        session_id: str
    ):

        result = (
            supabase
            .table("sessions")
            .select("id")
            .eq(
                "id",
                session_id
            )
            .eq(
                "user_id",
                self.user_id
            )
            .execute()
        )

        return len(result.data) > 0

    # =====================================
    # GET SESSION
    # =====================================

    def get_session(
        self,
        session_id: str
    ):

        return self.load_session(
            session_id
        )
    # =====================================
    # LIST SESSIONS
    # =====================================

    def list_sessions(
        self
    ):

        result = (
            supabase
            .table("sessions")
            .select("*")
            .eq(
                "user_id",
                self.user_id
            )
            .order(
                "created_at",
                desc=True
            )
            .execute()
        )

        sessions = []

        for row in result.data:

            sessions.append(
                {

                    "session_id": row["id"],

                    "title": row.get(
                        "title",
                        "New Chat"
                    ),

                    "created_at": row.get(
                        "created_at",
                        ""
                    )
                }
            )

        return sessions

    # =====================================
    # UPDATE TITLE
    # =====================================

    def update_title(
        self,
        session_id: str,
        new_title: str
    ):

        (
            supabase
            .table("sessions")
            .update(
                {
                    "title": new_title
                }
            )
            .eq(
                "id",
                session_id
            )
            .eq(
                "user_id",
                self.user_id
            )
            .execute()
        )

        return True

    # =====================================
    # UPDATE SUMMARY
    # =====================================

    def update_summary(
        self,
        session_id: str,
        summary: str
    ):

        (
            supabase
            .table("sessions")
            .update(
                {
                    "summary": summary
                }
            )
            .eq(
                "id",
                session_id
            )
            .eq(
                "user_id",
                self.user_id
            )
            .execute()
        )

        return True

    # =====================================
    # GET HISTORY
    # =====================================

    def get_history(
        self,
        session_id: str
    ):

        result = (
            supabase
            .table("messages")
            .select("*")
            .eq(
                "session_id",
                session_id
            )
            .order(
                "created_at"
            )
            .execute()
        )

        history = []

        for row in result.data:

            history.append(
                {
                    "role": row["role"],
                    "content": row["content"]
                }
            )

        return history

    # =====================================
    # SAVE HISTORY
    # =====================================

    def save_history(
        self,
        session_id: str,
        history: list
    ):

        # Delete old history

        (
            supabase
            .table("messages")
            .delete()
            .eq(
                "session_id",
                session_id
            )
            .execute()
        )

        if not history:

            return

        rows = []

        for message in history:

            rows.append(
                {
                    "session_id": session_id,
                    "role": message.get(
                        "role",
                        "assistant"
                    ),
                    "content": message.get(
                        "content",
                        ""
                    )
                }
            )

        (
            supabase
            .table("messages")
            .insert(
                rows
            )
            .execute()
        )
    # =====================================
    # DELETE SESSION
    # =====================================

    def delete_session(
        self,
        session_id: str
    ):

        # Delete selected PDFs

        (
            supabase
            .table("selected_pdfs")
            .delete()
            .eq(
                "session_id",
                session_id
            )
            .execute()
        )

        # Delete PDF metadata

        (
            supabase
            .table("pdfs")
            .delete()
            .eq(
                "session_id",
                session_id
            )
            .execute()
        )

        # Delete messages

        (
            supabase
            .table("messages")
            .delete()
            .eq(
                "session_id",
                session_id
            )
            .execute()
        )

        # Delete session

        (
            supabase
            .table("sessions")
            .delete()
            .eq(
                "id",
                session_id
            )
            .eq(
                "user_id",
                self.user_id
            )
            .execute()
        )

        return True

    # =====================================
    # ADD PDF
    # =====================================

    def add_pdf(
        self,
        session_id: str,
        pdf_name: str,
        pdf_path: str
    ):

        existing = (
            supabase
            .table("pdfs")
            .select("id")
            .eq(
                "session_id",
                session_id
            )
            .eq(
                "pdf_name",
                pdf_name
            )
            .execute()
        )

        if existing.data:

            return True

        (
            supabase
            .table("pdfs")
            .insert(
                {
                    "session_id": session_id,
                    "pdf_name": pdf_name,
                    "pdf_path": pdf_path
                }
            )
            .execute()
        )

        return True

    # =====================================
    # REMOVE PDF
    # =====================================

    def remove_pdf(
        self,
        session_id: str,
        pdf_name: str
    ):

        (
            supabase
            .table("selected_pdfs")
            .delete()
            .eq(
                "session_id",
                session_id
            )
            .eq(
                "pdf_name",
                pdf_name
            )
            .execute()
        )

        (
            supabase
            .table("pdfs")
            .delete()
            .eq(
                "session_id",
                session_id
            )
            .eq(
                "pdf_name",
                pdf_name
            )
            .execute()
        )

        return True

    # =====================================
    # GET PDFS
    # =====================================

    def get_pdfs(
        self,
        session_id: str
    ):

        result = (
            supabase
            .table("pdfs")
            .select("*")
            .eq(
                "session_id",
                session_id
            )
            .order(
                "uploaded_at"
            )
            .execute()
        )

        pdfs = []

        for row in result.data:

            pdfs.append(
                {
                    "name": row["pdf_name"],
                    "path": row["pdf_path"],
                    "uploaded_at": row[
                        "uploaded_at"
                    ]
                }
            )

        return pdfs
    # =====================================
    # SAVE SELECTED PDFS
    # =====================================

    def save_selected_pdfs(
        self,
        session_id: str,
        selected_pdfs: list
    ):

        # Remove previous selection

        (
            supabase
            .table("selected_pdfs")
            .delete()
            .eq(
                "session_id",
                session_id
            )
            .execute()
        )

        if not selected_pdfs:

            return True

        rows = []

        for pdf_name in selected_pdfs:

            rows.append(
                {
                    "session_id": session_id,
                    "pdf_name": pdf_name
                }
            )

        (
            supabase
            .table("selected_pdfs")
            .insert(
                rows
            )
            .execute()
        )

        return True

    # =====================================
    # GET SELECTED PDFS
    # =====================================

    def get_selected_pdfs(
        self,
        session_id: str
    ):

        result = (
            supabase
            .table("selected_pdfs")
            .select("pdf_name")
            .eq(
                "session_id",
                session_id
            )
            .execute()
        )

        return [
            row["pdf_name"]
            for row in result.data
        ]

    # =====================================
    # CLEAR SELECTED PDFS
    # =====================================

    def clear_selected_pdfs(
        self,
        session_id: str
    ):

        (
            supabase
            .table("selected_pdfs")
            .delete()
            .eq(
                "session_id",
                session_id
            )
            .execute()
        )

        return True

    # =====================================
    # UPDATE SESSION
    # =====================================

    def update_session(
        self,
        session_id: str,
        data: dict
    ):

        (
            supabase
            .table("sessions")
            .update(
                {
                    "title": data.get(
                        "title",
                        "New Chat"
                    ),

                    "summary": data.get(
                        "summary",
                        ""
                    )
                }
            )
            .eq(
                "id",
                session_id
            )
            .eq(
                "user_id",
                self.user_id
            )
            .execute()
        )

        if "history" in data:

            self.save_history(
                session_id,
                data["history"]
            )

        if "selected_pdfs" in data:

            self.save_selected_pdfs(
                session_id,
                data["selected_pdfs"]
            )

        return True

    # =====================================
    # GET PDF PATHS
    # =====================================

    def get_pdf_paths(
        self,
        session_id: str
    ):

        pdfs = self.get_pdfs(
            session_id
        )

        return [
            pdf["path"]
            for pdf in pdfs
        ]
