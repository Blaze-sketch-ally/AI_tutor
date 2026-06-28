from storage.storage_manager import StorageManager

storage = StorageManager()

session_id = "test_session"

pdf_path = "sample.pdf"      # Put any small PDF in your project folder

with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

print("\n===== UPLOADING =====")

result = storage.upload_pdf(
    session_id=session_id,
    pdf_name="sample.pdf",
    pdf_bytes=pdf_bytes
)

print(result)

print("\n===== LIST FILES =====")

files = storage.list_pdfs(session_id)

print(files)

print("\n===== DOWNLOADING =====")

download = storage.download_pdf(
    "test_session/sample.pdf"
)

print(download)

print("\n===== EXISTS =====")

print(
    storage.pdf_exists(
        session_id,
        "sample.pdf"
    )
)

print("\n===== DELETE =====")

print(
    storage.delete_pdf(
        "test_session/sample.pdf"
    )
)

print("\n===== LIST AFTER DELETE =====")

print(
    storage.list_pdfs(
        session_id
    )
)