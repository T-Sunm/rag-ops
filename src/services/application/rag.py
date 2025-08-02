from src.cache.semantic_cache import semantic_cache_llms
from src.services.domain.generator import GeneratorService
from src.services.domain.summarize import SummarizeService
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from src.config.settings import SETTINGS
from src.infrastructure.vector_stores.chroma_client import ChromaClientService
from src.schemas.domain.retrieval import SearchArgs

from langfuse import observe
from langfuse.langchain import CallbackHandler
from langfuse import get_client
import uuid
from nemoguardrails import LLMRails


class Rag:
    def __init__(self):
        self.llm = ChatOpenAI(**SETTINGS.llm_config)
        self.chroma_client = ChromaClientService()
        self.langfuse_handler = CallbackHandler()
        self.langfuse = get_client()

        # In-memory storage cho session history
        self.session_histories: dict[str, list[dict]] = {}

        # Define search tool
        self.search_tool = StructuredTool.from_function(
            name="search_docs",
            description=(
                "Retrieve documents from Chroma.\n"
                "Args:\n"
                "    question (str): the question.\n"
                "    top_k (int): the number of documents to retrieve.\n"
                "    with_score (bool): whether to include similarity scores.\n"
                "    metadata_filter (dict): filter by metadata.\n"
            ),
            func=self.chroma_client.retrieve_vector,
            args_schema=SearchArgs,
        )

        # Define tools dictionary
        self.tools = {"search_docs": self.search_tool}

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(list(self.tools.values()))

        # Initialize services
        self.generator_service = GeneratorService(
            llm_with_tools=self.llm_with_tools,
            tools=self.tools,
            langfuse_handler=self.langfuse_handler,
        )

        self.summarize_service = SummarizeService(
            langfuse_handler=self.langfuse_handler,
        )

    def _get_session_history(self, session_id: str | None = None) -> list[dict]:
        """Lấy chat history từ in-memory storage"""
        if not session_id:
            return []

        return self.session_histories.get(session_id, [])

    def _save_to_session_history(self, session_id: str, question: str, response: str):
        """Lưu vào in-memory storage"""
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []

        # Add user message và assistant response
        self.session_histories[session_id].extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]
        )

    @semantic_cache_llms.cache(namespace="pre-cache")
    @observe(name="get_response")
    async def get_response(
        self,
        question: str,
        session_id: str | None = None,
        user_id: str | None = None,
        guardrails: LLMRails | None = None,
    ):
        # ———— ID Normalization ————
        session_id = session_id or str(uuid.uuid4())
        user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"

        # ———— History Retrieval & Summarization ————
        chat_history = self._get_session_history(session_id)
        if len(chat_history) >= 6:
            chat_history = await self.summarize_service._summarize_and_truncate_history(
                chat_history,
                keep_last=4,
                session_id=session_id,
                user_id=user_id,
            )
            # cập nhật lại history đã rút gọn
            self.session_histories[session_id] = chat_history

        # ———— Nếu có Guardrails thì dùng nó ————
        if guardrails:
            messages = [
                {
                    "role": "context",
                    "content": {"session_id": session_id, "user_id": user_id},
                },
                {"role": "user", "content": question},
            ]
            # Guardrails tự động chạy input→dialog→output rails
            result = await guardrails.generate_async(
                prompt=messages,
                options={"metadata": {"session_id": session_id, "user_id": user_id}},
            )

            response_str = str(result.response)
            # Không cần lưu history nếu Guardrails block ; Nếu guardrails ok thì lưu
            if "sorry" not in response_str.lower():
                self._save_to_session_history(session_id, question, response_str)
            return {
                "response": response_str,
                "session_id": session_id,
                "user_id": user_id,
            }

        # ———— Fallback: chạy RAG thường ————

        rag_output = await self.generator_service.generate(
            question=question,
            chat_history=chat_history,
            session_id=session_id,
            user_id=user_id,
        )

        # lưu lại history sau khi RAG trả về
        self._save_to_session_history(session_id, question, rag_output)
        return {"response": rag_output, "session_id": session_id, "user_id": user_id}

    async def get_sse_response(self, question: str, session_id: str, user_id: str):
        # Lấy history từ memory
        chat_history = self._get_session_history(session_id)
        if len(chat_history) >= 6:
            chat_history = await self.summarize_service._summarize_and_truncate_history(
                chat_history, 4
            )
            self.session_histories[session_id] = chat_history

        # Collect full response để save sau
        full_response = ""
        async for message in self.generator_service.generate_stream(
            question=question,
            chat_history=chat_history.copy(),
            session_id=session_id,
            user_id=user_id,
        ):
            full_response += message
            yield f"event: responseUpdate\ndata: {message}\n\n"

        # Save conversation sau khi stream xong
        self._save_to_session_history(session_id, question, full_response)

        # Kết thúc stream
        yield "event: responseUpdate\ndata: [DONE]\n\n"


rag_service = Rag()
