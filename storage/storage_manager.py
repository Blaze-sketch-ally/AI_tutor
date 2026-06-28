import os
import tempfile

from database.supabase_client import supabase


class StorageManager:

    def __init__(self):

        self.bucket = "pdfs"

    # =====================================
    # UPLOAD PDF
    # =====================================

    def upload_pdf(
        self,
        session_id: str,
        pdf_name: str,
        pdf_bytes: bytes
    ):

        try:

            storage_path = (
                f"{session_id}/{pdf_name}"
            )

            response = (
                supabase.storage
                .from_(self.bucket)
                .upload(
                    path=storage_path,
                    file=bytes(pdf_bytes),
                    file_options={
                        "content-type": "application/pdf",
                        "upsert": "true"
                    }
                )
            )

            return {

                "success": True,

                "storage_path": storage_path,

                "response": response

            }

        except Exception as e:

            print(
                f"Upload Error: {e}"
            )

            return {

                "success": False,

                "message": str(e)

            }

    # =====================================
    # DOWNLOAD PDF
    # =====================================

    def download_pdf(
        self,
        storage_path: str
    ):

        try:

            pdf_bytes = (
                supabase.storage
                .from_(self.bucket)
                .download(
                    storage_path
                )
            )

            temp_dir = tempfile.gettempdir()

            filename = os.path.basename(
                storage_path
            )

            temp_path = os.path.join(
                temp_dir,
                filename
            )

            with open(
                temp_path,
                "wb"
            ) as f:

                f.write(
                    pdf_bytes
                )

            return {

                "success": True,

                "temp_path": temp_path

            }

        except Exception as e:

            print(
                f"Download Error: {e}"
            )

            return {

                "success": False,

                "message": str(e)

            }

    # =====================================
    # DELETE PDF
    # =====================================

    def delete_pdf(
        self,
        storage_path: str
    ):

        try:

            supabase.storage.from_(
                self.bucket
            ).remove(
                [storage_path]
            )

            return True

        except Exception as e:

            print(
                f"Delete Error: {e}"
            )

            return False

    # =====================================
    # CHECK PDF EXISTS
    # =====================================

    def pdf_exists(
        self,
        session_id: str,
        pdf_name: str
    ):

        try:

            files = (
                supabase.storage
                .from_(self.bucket)
                .list(
                    session_id
                )
            )

            for file in files:

                if (
                    file["name"]
                    == pdf_name
                ):

                    return True

            return False

        except Exception:

            return False

    # =====================================
    # LIST PDFs
    # =====================================

    def list_pdfs(
        self,
        session_id: str
    ):

        try:

            files = (
                supabase.storage
                .from_(self.bucket)
                .list(
                    session_id
                )
            )

            return files

        except Exception as e:

            print(
                f"List Error: {e}"
            )

            return []