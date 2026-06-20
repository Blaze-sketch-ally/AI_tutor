import os
import uuid
import streamlit as st

from agents.orchestrator import Orchestrator
from rag.ingest import PDFIngestor

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Tutor Agent",
    page_icon="🧠",
    layout="wide"
)

# ==================================================
# SESSION STATE
# ==================================================

if "session_id" not in st.session_state:

    st.session_state.session_id = str(
        uuid.uuid4()
    )

if "last_uploaded_pdf" not in st.session_state:

    st.session_state.last_uploaded_pdf = None

if "messages" not in st.session_state:

    st.session_state.messages = []

# ==================================================
# LOAD ORCHESTRATOR
# ==================================================

@st.cache_resource
def load_orchestrator(session_id):

    return Orchestrator(
        session_id=session_id
    )

orchestrator = load_orchestrator(
    st.session_state.session_id
)

# ==================================================
# HEADER
# ==================================================

st.title("🧠 AI Tutor Agent")

st.markdown(
    """
Upload PDFs and chat naturally.

### Features

* Conversational AI Tutor
* PDF Question Answering
* Hybrid RAG Search
* Quiz Generation
* Notes Generation
* Diagram Generation
* YouTube Recommendations
* Research Papers
* Session-based Memory
"""
)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("📂 Upload PDF")

st.sidebar.write(
    f"Session: {st.session_state.session_id[:8]}"
)

st.sidebar.markdown("---")

# ==================================================
# NEW SESSION
# ==================================================

if st.sidebar.button(
    "🔄 Start New Session"
):

    st.cache_resource.clear()

    st.session_state.session_id = str(
        uuid.uuid4()
    )

    st.session_state.last_uploaded_pdf = None

    st.session_state.messages = []

    st.rerun()

# ==================================================
# PDF UPLOAD
# ==================================================

uploaded_file = st.sidebar.file_uploader(
    "Choose PDF",
    type=["pdf"]
)

if (
    uploaded_file
    and uploaded_file.name
    != st.session_state.last_uploaded_pdf
):

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    save_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(save_path, "wb") as f:

        f.write(
            uploaded_file.getbuffer()
        )

    with st.spinner(
        "Indexing PDF..."
    ):

        ingestor = PDFIngestor(
            session_id=st.session_state.session_id
        )

        result = ingestor.ingest_pdf(
            save_path
        )

    if result["status"] == "success":

        st.session_state.last_uploaded_pdf = (
            uploaded_file.name
        )

        st.sidebar.success(
            f"Indexed {result['chunks_created']} chunks"
        )

    else:

        st.sidebar.error(
            result["message"]
        )

# ==================================================
# CHAT HISTORY
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# ==================================================
# CHAT INPUT
# ==================================================

query = st.chat_input(
    "Ask anything..."
)

# ==================================================
# RUN PIPELINE
# ==================================================

if query:

    # -----------------------------
    # USER MESSAGE
    # -----------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    # -----------------------------
    # ASSISTANT RESPONSE
    # -----------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            result = orchestrator.run(
                query=query
            )

        assistant_text = ""

        # =====================================
        # CHAT
        # =====================================

        if "chat" in result:

            assistant_text += (
                result["chat"]
                + "\n\n"
            )

        # =====================================
        # ANSWER
        # =====================================

        if "answer" in result:

            assistant_text += (
                result["answer"]
                + "\n\n"
            )

        # =====================================
        # NOTES
        # =====================================

        if "notes" in result:

            assistant_text += (
                "## 📝 Notes\n\n"
                + result["notes"]
                + "\n\n"
            )

        # =====================================
        # QUIZ
        # =====================================

        if "quiz" in result:

            assistant_text += (
                "## ❓ Quiz\n\n"
                + result["quiz"]
                + "\n\n"
            )

        st.markdown(
            assistant_text
        )

        # =====================================
        # DIAGRAM
        # =====================================

        if "diagram" in result:

            st.subheader(
                "📊 Diagram"
            )

            st.code(
                result["diagram"],
                language="mermaid"
            )

        # =====================================
        # VIDEOS
        # =====================================

        if (
            "videos" in result
            and result["videos"]
        ):

            st.subheader(
                "📺 YouTube Videos"
            )

            for video in result["videos"]:

                if not isinstance(
                    video,
                    dict
                ):
                    continue

                st.markdown(
                    f"### {video.get('title', 'Untitled Video')}"
                )

                if video.get(
                    "channel"
                ):

                    st.write(
                        f"**Channel:** {video['channel']}"
                    )

                if video.get(
                    "url"
                ):

                    st.link_button(
                        "▶ Watch Video",
                        video["url"]
                    )

        # =====================================
        # PAPERS
        # =====================================

        if (
            "papers" in result
            and result["papers"]
        ):

            st.subheader(
                "📄 Research Papers"
            )

            for paper in result["papers"]:

                if not isinstance(
                    paper,
                    dict
                ):
                    continue

                st.markdown(
                    f"### {paper.get('title', 'Untitled Paper')}"
                )

                if paper.get(
                    "authors"
                ):

                    st.write(
                        "**Authors:**",
                        ", ".join(
                            paper["authors"]
                        )
                    )

                if paper.get(
                    "published"
                ):

                    st.write(
                        f"**Published:** {paper['published']}"
                    )

                if paper.get(
                    "summary"
                ):

                    with st.expander(
                        "Abstract"
                    ):

                        st.write(
                            paper["summary"]
                        )

                if paper.get(
                    "pdf_url"
                ):

                    st.link_button(
                        "📄 Open Paper",
                        paper["pdf_url"]
                    )

    # -----------------------------
    # SAVE ASSISTANT MESSAGE
    # -----------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_text
        }
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Powered by Groq + LangChain + LangSmith + ChromaDB + YouTube + arXiv | Made by Blaze"
)