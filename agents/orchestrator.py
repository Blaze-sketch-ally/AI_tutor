from langsmith import traceable

from agents.chat_agent import ChatAgent
from agents.tutor_agent import TutorAgent
from agents.youtube_agent import YouTubeAgent
from agents.paper_agent import PaperAgent
from agents.quiz_agent import QuizAgent
from agents.notes_agent import NotesAgent
from agents.diagram_agent import DiagramAgent
from agents.intent_classifier import IntentClassifier
from agents.topic_extractor import TopicExtractor
from agents.query_resolver import QueryResolverAgent
from agents.comparison_agent import ComparisonAgent
from agents.cross_pdf_notes_agent import CrossPDFNotesAgent
from agents.cross_pdf_quiz_agent import CrossPDFQuizAgent

from memory.chat_memory import ChatMemory
from memory.session_manager import SessionManager
from memory.conversation_summary import (
    ConversationSummaryAgent
)

from rag.retriever import Retriever


class Orchestrator:

    def __init__(
        self,
        username:str,
        session_id: str
    ):

        self.session_id = session_id
        self.username=username

        self.retriever = Retriever(
            session_id=session_id
        )

        self.memory = ChatMemory(
            username=username,
            session_id=session_id
        )
        self.session_manager = SessionManager(
            username=username
        )

        self.chat = ChatAgent()
        self.tutor = TutorAgent()

        self.youtube = YouTubeAgent()
        self.paper = PaperAgent()

        self.quiz = QuizAgent()
        self.notes = NotesAgent()
        self.diagram = DiagramAgent()

        self.intent_classifier = (
            IntentClassifier()
        )

        self.topic_extractor = (
            TopicExtractor()
        )

        self.summary_agent = (
            ConversationSummaryAgent()
        )

        self.query_resolver = (
            QueryResolverAgent()
        )
        self.comparison = ComparisonAgent()
        self.cross_notes = (
                        CrossPDFNotesAgent()
        )

        self.cross_quiz = (
                        CrossPDFQuizAgent()
        )   

    @traceable(
        name="Orchestrator.run_full_pipeline"
    )
    def run(
        self,
        query: str,
        marks: str = "8",
        selected_pdfs:list[str]|None=None
    ):

        # =====================================
        # HISTORY BEFORE CURRENT QUERY
        # =====================================

        previous_history = (
            self.memory.get_history_text()
        )

        # =====================================
        # CONVERSATION SUMMARY
        # =====================================

        conversation_summary = (
            self.summary_agent.summarize(
                history_text=previous_history,
                current_query=query
            )
        )

        # =====================================
        # QUERY RESOLUTION
        # =====================================

        resolver_result = (
            self.query_resolver.resolve(
                query=query,
                history_text=previous_history,
                summary=conversation_summary
            )
        )

        resolved_query = (
            resolver_result.get(
                "resolved_query",
                query
            )
        )

        # =====================================
        # SAVE USER MESSAGE
        # =====================================

        self.memory.add_message(
            "user",
            query
        )
        if not self.session_manager.session_exists(
            self.session_id
        ):
            self.session_manager.create_session(
                title="New Chat"
            )
        if self.memory.size() == 1:

            self.session_manager.update_title(
                self.session_id,
                query[:50]
            )

        history_text = (
            self.memory.get_history_text()
        )

        # =====================================
        # INTENT CLASSIFICATION
        # =====================================

        intents = (
            self.intent_classifier.classify(
                resolved_query
            )
        )
        print("\n========== INTENTS ==========")
        print(intents)
        print("=============================\n")

        knowledge_source = intents.get(
            "knowledge_source",
            "general"
        )

        # =====================================
        # PDF RETRIEVAL + SOURCES
        # =====================================

        pdf_context = ""
        pdf_contexts={}
        sources = []
        strict_pdf = False

        try:

            if (
                (
                    knowledge_source in [
                        "pdf",
                        "hybrid"
                    ]
                    or intents.get(
                        "comparison",
                        False
                    )
                )
                and self.retriever.has_documents()
            ):

    # =====================================
    # CROSS PDF RETRIEVAL
    # =====================================

                if (
                    intents.get("all_pdfs", False)
                    or intents.get("comparison", False)
                ):

                    grouped_result = (
                        self.retriever.get_grouped_context_by_source(
                            resolved_query,
                            selected_pdfs=selected_pdfs,
                            top_k=5
                        )
                    )

                    pdf_contexts = (
                        grouped_result.get(
                            "pdf_contexts",
                            {}
                        )
                    )
                    print("\n===== PDF CONTEXTS =====")
                    print(pdf_contexts.keys())
                    print("========================\n")

                    sources = (
                        grouped_result.get(
                            "sources",
                            []
                        )
                    )

    # =====================================
    # NORMAL RETRIEVAL
    # =====================================

                else:

                    retrieval_result = (
                        self.retriever.get_context_with_sources(
                            query=resolved_query,
                            selected_pdfs=selected_pdfs,
                            top_k=5
                        )
                    )

                    pdf_context = (
                        retrieval_result.get(
                            "context",
                            ""
                        )
                    )

                    sources = (
                        retrieval_result.get(
                            "sources",
                            []
                        )
                    )

            if knowledge_source == "pdf":

                strict_pdf = True
        except Exception as e:

            print(
                f"Retriever Error: {e}"
            )

            pdf_context = ""
            pdf_contexts={}
            sources = []

        has_pdf = bool(
            pdf_context.strip()
        ) or bool(
            pdf_contexts
    )
        # =====================================
        # DEBUG
        # =====================================

        print("\n========== DEBUG ==========")

        print(
            "Session:",
            self.session_id
        )

        print(
            "Query:",
            query
        )

        print(
            "Resolved Query:",
            resolved_query
        )

        print(
            "Follow Up:",
            resolver_result.get(
                "is_followup",
                False
            )
        )

        print(
            "Resolved Topic:",
            resolver_result.get(
                "resolved_topic",
                ""
            )
        )

        print(
            "Intents:",
            intents
        )

        print(
            "Knowledge Source:",
            knowledge_source
        )
        print(
                "Selected PDFs:",
                selected_pdfs
            )

        print(
            "PDF Context Found:",
            has_pdf
        )

        print(
            "Sources Found:",
            len(sources)
        )

        print(
            "Memory Size:",
            self.memory.size()
        )

        print(
            "Summary Generated:",
            bool(conversation_summary)
        )

        print("===========================\n")

        # =====================================
        # PURE CHAT MODE
        # =====================================

        is_only_chat = (

            intents.get("chat", False)

            and not intents.get(
                "answer",
                False
            )

            and not intents.get(
                "quiz",
                False
            )
            and not intents.get(
                "comparison",
                False
            )

            and not intents.get(
                "notes",
                False
            )

            and not intents.get(
                "diagram",
                False
            )

            and not intents.get(
                "videos",
                False
            )

            and not intents.get(
                "papers",
                False
            )

        )

        if is_only_chat:

            reply = (
                self.chat.generate_reply(
                    message=query,
                    history_text=history_text,
                    summary=conversation_summary
                )
            )

            self.memory.add_message(
                message={
                    "role": "assistant",
                    "content": memory_text,
                    "answer": response.get("answer"),
                    "notes": response.get("notes"),
                    "quiz": response.get("quiz"),
                    "comparison": response.get("comparison"),
                    "diagram": response.get("diagram"),
                    "videos": response.get("videos", []),
                    "papers": response.get("papers", []),
                    "sources": response.get("sources", [])
                }
            )

            return {
                "mode": "chat",
                "chat": reply,
                "sources": []
            }

        # =====================================
        # RESPONSE OBJECT
        # =====================================

        response = {
            "mode": "mixed",
            "sources": sources
        }

        # =====================================
        # ANSWER
        # =====================================

        if intents.get(
            "answer",
            False
        ):

            response["answer"] = (
                self.tutor.generate_answer(
                    question=query,
                    context=pdf_context,
                    history_text=history_text,
                    summary=conversation_summary,
                    marks=marks,
                    strict_pdf=strict_pdf
                )
            )

        # =====================================
        # NOTES
        # =====================================

        if intents.get(
            "notes",
            False
    ):

            if intents.get(
                "all_pdfs",
                False
            ):

                print("Cross PDF Notes PDFs:")
                print(pdf_contexts.keys())
                response["notes"] = (
                    self.cross_notes.generate_notes(
                        topic=query,
                        pdf_contexts=pdf_contexts
                    )
                )

            else:

                response["notes"] = (
                    self.notes.generate_notes(
                        topic=query,
                        context=pdf_context,
                        history_text=history_text,
                        summary=conversation_summary,
                        strict_pdf=strict_pdf
                    )
                )
        # =====================================
# COMPARISON
# =====================================

        if intents.get(
            "comparison",
            False
        ):

            response["comparison"] = (
                self.comparison.generate_comparison(
                    topic=query,
                    pdf_contexts=pdf_contexts,
                    history_text=history_text,
                    summary=conversation_summary
                )
            )

        # =====================================
        # QUIZ
        # =====================================

        if intents.get(
    "quiz",
    False
):

            if intents.get(
                "all_pdfs",
                False
            ):

                print("Cross PDF Quiz PDFs:")
                print(pdf_contexts.keys())
                response["quiz"] = (
                    self.cross_quiz.generate_quiz(
                        topic=query,
                        pdf_contexts=pdf_contexts
                    )
                )

            else:

                response["quiz"] = (
                    self.quiz.generate_quiz(
                        topic=query,
                        context=pdf_context,
                        history_text=history_text,
                        summary=conversation_summary,
                        strict_pdf=strict_pdf
                    )
                )

        # =====================================
        # DIAGRAM
        # =====================================

        if intents.get(
            "diagram",
            False
        ):

            response["diagram"] = (
                self.diagram.generate_diagram(
                    topic=query,
                    context=pdf_context,
                    history_text=history_text,
                    summary=conversation_summary,
                    strict_pdf=strict_pdf
                )
            )

        # =====================================
        # VIDEOS
        # =====================================

        if intents.get(
            "videos",
            False
        ):

            response["videos"] = (
                self.youtube.get_videos(
                    query=query,
                    context=pdf_context,
                    history_text=history_text,
                    summary=conversation_summary
                )
            )

        # =====================================
        # PAPERS
        # =====================================

        if intents.get(
            "papers",
            False
        ):

            response["papers"] = (
                self.paper.get_papers(
                    query=query,
                    context=pdf_context,
                    history_text=history_text,
                    summary=conversation_summary
                )
            )

        # =====================================
        # FALLBACK ANSWER
        # =====================================

        if not any(
                key in response
                for key in [
                    "answer",
                    "comparison",
                    "notes",
                    "quiz",
                    "diagram",
                    "videos",
                    "papers"
                ]
            ):

            response["answer"] = (
                self.tutor.generate_answer(
                    question=query,
                    context=pdf_context,
                    history_text=history_text,
                    summary=conversation_summary,
                    marks=marks,
                    strict_pdf=strict_pdf
                )
            )

        # =====================================
        # SAVE ASSISTANT RESPONSE
        # =====================================

        memory_text = ""

        if "answer" in response:
            memory_text += response["answer"]

        if "comparison" in response:
            memory_text += "\n\n" + response["comparison"]

        if "notes" in response:
            memory_text += "\n\n" + response["notes"]

        if "quiz" in response:
            memory_text += "\n\n" + response["quiz"]

        if "diagram" in response:
            memory_text += "\n\n" + response["diagram"]

        if memory_text.strip():

            self.memory.add_message(
                "assistant",
                memory_text
            )
        self.session_manager.save_history(
            self.session_id,
            self.memory.history
        )

        return response
                