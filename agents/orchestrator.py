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

from memory.chat_memory import ChatMemory
from memory.conversation_summary import (
    ConversationSummaryAgent
)

from rag.retriever import Retriever


class Orchestrator:

    def __init__(
        self,
        session_id: str
    ):

        self.session_id = session_id

        self.retriever = Retriever(
            session_id=session_id
        )

        self.memory = ChatMemory(
            session_id=session_id
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

    @traceable(
        name="Orchestrator.run_full_pipeline"
    )
    def run(
        self,
        query: str,
        marks: str = "8"
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

        knowledge_source = intents.get(
            "knowledge_source",
            "general"
        )

        # =====================================
        # PDF RETRIEVAL
        # =====================================

        pdf_context = ""
        strict_pdf = False

        try:

            if (
                knowledge_source in [
                    "pdf",
                    "hybrid"
                ]
                and self.retriever.has_documents()
            ):

                pdf_context = (
                    self.retriever.get_context(
                        resolved_query
                    )
                )

                if knowledge_source == "pdf":

                    strict_pdf = True

        except Exception as e:

            print(
                f"Retriever Error: {e}"
            )

            pdf_context = ""

        has_pdf = bool(
            pdf_context.strip()
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
            "PDF Context Found:",
            has_pdf
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
                "assistant",
                reply
            )

            return {
                "mode": "chat",
                "chat": reply
            }

        # =====================================
        # RESPONSE OBJECT
        # =====================================

        response = {
            "mode": "mixed"
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
        # QUIZ
        # =====================================

        if intents.get(
            "quiz",
            False
        ):

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

        if len(response) == 1:

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

            memory_text += (
                response["answer"]
            )

        if "notes" in response:

            memory_text += (
                "\n\n" +
                response["notes"]
            )

        if "quiz" in response:

            memory_text += (
                "\n\n" +
                response["quiz"]
            )

        if "diagram" in response:

            memory_text += (
                "\n\n" +
                response["diagram"]
            )

        if memory_text.strip():

            self.memory.add_message(
                "assistant",
                memory_text
            )

        return response