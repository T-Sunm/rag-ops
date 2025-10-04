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
from logging import getLogger

logger = getLogger(__name__)


class Rag:
    def __init__(self):
        self.llm = ChatOpenAI(**SETTINGS.llm_config)
        self.chroma_client = ChromaClientService()
        self.langfuse_handler = CallbackHandler()
        self.langfuse = get_client()

        # Không cần in-memory storage nữa vì sẽ lấy từ Langfuse
        # self.session_histories: dict[str, list[dict]] = {}

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

    def get_session_history(self, session_id: str | None = None) -> list[dict]:
        """Lấy chat history từ Langfuse dựa trên cấu trúc trace thực tế."""
        if not session_id:
            return []

        try:
            # Lấy traces từ Langfuse
            traces_in_session = self.langfuse.api.trace.list(
                session_id=session_id, limit=100
            )

            # Sắp xếp theo thời gian
            sorted_traces = sorted(traces_in_session.data, key=lambda x: x.timestamp)
            chat_history = []

            for trace in sorted_traces:
                ai_answer = ""
                user_question = ""

                if isinstance(trace.output, str):
                    ai_answer = trace.output
                elif isinstance(trace.output, dict):
                    ai_answer = trace.output.get("content", "") or trace.output.get(
                        "response", ""
                    )

                if isinstance(trace.input, dict):
                    user_question = trace.input.get("question", "")

                if user_question and ai_answer:
                    chat_history.extend(
                        [
                            {"role": "user", "content": user_question},
                            {"role": "assistant", "content": ai_answer},
                        ]
                    )

            max_pairs = 6
            return chat_history[-(max_pairs * 2) :]

        except Exception as e:
            print(f"Error fetching chat history from Langfuse: {e}")
            return []

    @semantic_cache_llms.cache(namespace="pre-cache")
    async def get_response(
        self,
        question: str,
        session_id: str | None = None,
        user_id: str | None = None,
        guardrails: LLMRails | None = None,
    ):
        with self.langfuse.start_as_current_span(
            name="get_restapi_response",
            input={"question": question, "session_id": session_id, "user_id": user_id},
        ) as span:
            self.langfuse.update_current_trace(session_id=session_id, user_id=user_id)

            chat_history = self.get_session_history(session_id)
            print("chat_history is ", chat_history)

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
                # KHÔNG trace guardrails.generate_async để tránh lưu config phức tạp
                result = await guardrails.generate_async(prompt=messages)

                if is_guardrails_error(result):
                    blocked_response = "I'm sorry, but I cannot provide a response to that request. The content was blocked by our safety guidelines."
                    span.update(output=blocked_response)
                    return blocked_response

                response = str(result)
                span.update(output=response)
                return response

            # ———— Fallback: chạy RAG thường ————
            rag_output = await self.rest_generator_service.generate(
                question=question,
                chat_history=chat_history,
                session_id=session_id,
                user_id=user_id,
            )

            span.update(output=rag_output)
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
            chat_history = self.get_session_history(session_id)

            # ———— CHECK INPUT RAILS TRƯỚC KHI GỌI LLM ————
            if guardrails:
                messages = [
                    {
                        "role": "context",
                        "content": {"session_id": session_id, "user_id": user_id},
                    },
                    {"role": "user", "content": question},
                ]

                # Chỉ check input rails
                input_check_result = await guardrails.generate_async(
                    messages=messages,
                    options={"rails": ["input"]},  # CHỈ CHẠY INPUT RAILS
                )

                # Access the response attribute which contains the list of messages
                response_messages = input_check_result.response
                print(f"Response messages: {response_messages}")

                # Check if there's an assistant message indicating blocking
                assistant_message = None
                for msg in response_messages:
                    if msg.get("role") == "assistant":
                        assistant_message = msg.get("content")
                        break

                # Lấy câu trả lời mặc định khi bị chặn từ config để so sánh
                default_cant_respond = "I'm sorry, I can't respond to that."

                if assistant_message and default_cant_respond in assistant_message:
                    # Input bị block hoàn toàn
                    yield f"{json.dumps(assistant_message)}\n\n"
                    span.update(output="Request blocked by input guardrails")
                    return

                # Nếu không bị block, tìm user message để kiểm tra có bị alter không
                user_message_content = None
                for msg in response_messages:
                    if msg.get("role") == "user":
                        user_message_content = msg.get("content")
                        break

                # Input bị alter, dùng input đã được alter : Chỉ khi bật Private AI Integration thì mới dùng input đã được alter
                # Tham khảo tại: https://docs.nvidia.com/nemo/guardrails/latest/user-guides/community/privateai.html
                if user_message_content and user_message_content != question:
                    # Input đã bị thay đổi (altered), ví dụ PII masking
                    # Cập nhật `question` để dùng cho các bước sau
                    print(
                        f"Input altered from '{question}' to '{user_message_content}'"
                    )
                    question = user_message_content

            # Tạo async generator cho external LLM streaming
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
                    # self._save_to_session_history(session_id, question, full_response) # Removed as per new_code
                    span.update(output=full_response)
                    # Kiểm tra và tóm tắt lịch sử nếu cần (chạy sau khi response xong)
                    # current_history = self.get_session_history(session_id) # Removed as per new_code
                    # if len(current_history) >= 4: # Removed as per new_code
                    #     summarized_history = await self.summarize_service._summarize_and_truncate_history( # Removed as per new_code
                    #         chat_history=current_history, keep_last=2 # Removed as per new_code
                    #     ) # Removed as per new_code
                    #     self.session_histories[session_id] = summarized_history # Removed as per new_code
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
            # self._save_to_session_history(session_id, question, full_response) # Removed as per new_code
            span.update(output=full_response)
            # Kiểm tra và tóm tắt lịch sử nếu cần (chạy sau khi response xong)
            # current_history = self.get_session_history(session_id) # Removed as per new_code
            # if len(current_history) >= 4: # Removed as per new_code
            #     summarized_history = ( # Removed as per new_code
            #         await self.summarize_service._summarize_and_truncate_history( # Removed as per new_code
            #             chat_history=current_history, keep_last=2 # Removed as per new_code
            #         ) # Removed as per new_code
            #     ) # Removed as per new_code
            #     self.session_histories[session_id] = summarized_history # Removed as per new_code


rag_service = Rag()
