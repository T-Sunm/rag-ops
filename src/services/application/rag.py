from src.cache.semantic_cache import semantic_cache_llms
from src.services.domain.generator import RestApiGeneratorService, SSEGeneratorService
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
import json
from src.utils.text_processing import is_guardrails_error


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
                "    query (str): the query.\n"
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
        self.rest_generator_service = RestApiGeneratorService(
            llm_with_tools=self.llm_with_tools,
            tools=self.tools,
            langfuse_handler=self.langfuse_handler,
        )
        self.sse_generator_service = SSEGeneratorService(
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
        chat_history = self._get_session_history(session_id)

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
            result = await guardrails.generate_async(prompt=messages)

            if is_guardrails_error(result):
                blocked_response = "I'm sorry, but I cannot provide a response to that request. The content was blocked by our safety guidelines."
                return blocked_response

            # Không cần lưu history nếu Guardrails block ; Nếu guardrails ok thì lưu
            self._save_to_session_history(session_id, question, str(result))
            # Kiểm tra và tóm tắt lịch sử nếu cần (chạy sau khi response xong)
            current_history = self._get_session_history(session_id)
            if len(current_history) >= 4:
                summarized_history = (
                    await self.summarize_service._summarize_and_truncate_history(
                        chat_history=current_history, keep_last=2
                    )
                )
                self.session_histories[session_id] = summarized_history
            return str(result)

        # ———— Fallback: chạy RAG thường ————

        rag_output = await self.rest_generator_service.generate(
            question=question,
            chat_history=chat_history,
            session_id=session_id,
            user_id=user_id,
        )

        # lưu lại history sau khi RAG trả về
        self._save_to_session_history(session_id, question, rag_output)
        # Kiểm tra và tóm tắt lịch sử nếu cần (chạy sau khi response xong)
        current_history = self._get_session_history(session_id)
        if len(current_history) >= 4:
            summarized_history = (
                await self.summarize_service._summarize_and_truncate_history(
                    chat_history=current_history, keep_last=2
                )
            )
            self.session_histories[session_id] = summarized_history
        return rag_output

    # ----------------------------------------------SSE----------------------------------------------
    @semantic_cache_llms.cache(namespace="pre-cache")
    async def get_sse_response(
        self,
        question: str,
        session_id: str,
        user_id: str,
        guardrails: LLMRails | None = None,
    ):
        with self.langfuse.start_as_current_span(
            name="get_sse_response",
            input={"question": question, "session_id": session_id, "user_id": user_id},
        ) as span:
            self.langfuse.update_current_trace(session_id=session_id, user_id=user_id)
            chat_history = self._get_session_history(session_id)

            # Tạo async generator cho external LLM streaming
            @observe()
            async def rag_token_generator(question, chat_history, session_id, user_id):
                """External generator sử dụng generator_service để tạo tokens"""
                async for message in self.sse_generator_service.generate_stream(
                    question=question,
                    chat_history=chat_history.copy(),  # Xài copy để tránh không edit vào chat_history gốc, để mỗi req đến ta chỉ lưu response cuối cùng
                    session_id=session_id,
                    user_id=user_id,
                ):
                    yield message

            # ———— Nếu có Guardrails thì dùng external generator ————
            if guardrails:
                messages = [
                    {
                        "role": "context",
                        "content": {"session_id": session_id, "user_id": user_id},
                    },
                    {"role": "user", "content": question},
                ]

                is_blocked = False
                full_response = ""
                # Sử dụng external generator với guardrails
                async for chunk in guardrails.stream_async(
                    messages=messages,
                    generator=rag_token_generator(
                        question, chat_history, session_id, user_id
                    ),
                ):
                    full_response += chunk

                    # Check if this chunk indicates blocking
                    if is_guardrails_error(chunk):
                        is_blocked = True
                        # Send a clean error message instead
                        error_message = "I'm sorry, but I cannot provide a response to that request."
                        yield f"{json.dumps(error_message)}\n\n"
                        break
                    else:
                        yield f"{json.dumps(chunk)}\n\n"

                # Only save to history if not blocked
                if not is_blocked:
                    self._save_to_session_history(session_id, question, full_response)
                    span.update(output=full_response)
                    # Kiểm tra và tóm tắt lịch sử nếu cần (chạy sau khi response xong)
                    current_history = self._get_session_history(session_id)
                    if len(current_history) >= 4:
                        summarized_history = await self.summarize_service._summarize_and_truncate_history(
                            chat_history=current_history, keep_last=2
                        )
                        self.session_histories[session_id] = summarized_history
                else:
                    span.update(output="Request blocked by guardrails")
                return

            # ———— Nếu không có Guardrails, streaming trực tiếp ————
            full_response = ""
            async for message in rag_token_generator(
                question, chat_history, session_id, user_id
            ):
                full_response += message
                yield f"{json.dumps(message)}\n\n"

            # Save conversation sau khi stream xong
            self._save_to_session_history(session_id, question, full_response)
            span.update(output=full_response)
            # Kiểm tra và tóm tắt lịch sử nếu cần (chạy sau khi response xong)
            current_history = self._get_session_history(session_id)
            if len(current_history) >= 4:
                summarized_history = (
                    await self.summarize_service._summarize_and_truncate_history(
                        chat_history=current_history, keep_last=2
                    )
                )
                self.session_histories[session_id] = summarized_history


rag_service = Rag()
