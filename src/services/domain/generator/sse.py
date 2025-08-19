from .base import BaseGeneratorService
from src.utils import logger
from src.utils.text_processing import build_context
from langchain_core.messages import AIMessage, SystemMessage
from src.cache.semantic_cache import semantic_cache_llms


class SSEGeneratorService(BaseGeneratorService):
    """Generator service dành cho SSE với streaming response và RAG integration"""

    async def _initial_llm_call(
        self,
        question: str,
        chat_history: list[dict],
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        """Phase 1: Initial LLM call để kiểm tra tool calls"""
        self._update_trace_context(session_id, user_id)
        formatted_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in chat_history
        )
        prompt_template = self.prompt_userinput.get_langchain_prompt(
            question=question, chat_history=formatted_history
        )
        messages = [SystemMessage(content=prompt_template)]

        # Dùng astream_events để có tool call info
        async for event in self.llm_with_tools.astream_events(
            messages,
            version="v1",
            config={
                "callbacks": [self.langfuse_handler],
                "metadata": {"session_id": session_id, "user_id": user_id},
            },
        ):
            yield event, messages

    async def _create_message(
        self,
        question: str,
        chat_history: list[dict],
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        """SSE version: stream initial call và check tool calls on-the-fly"""
        tool_calls = []
        full_response_content = ""
        messages = []
        tool_call_detected = False
        # Phase 1: Stream và parse events
        async for event, prompt_messages in self._initial_llm_call(
            question, chat_history, session_id, user_id
        ):
            messages = prompt_messages
            kind = event["event"]
            # Check on the fly if chunk is a tool call or a response
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                # Lấy content text nếu có
                if chunk.content and not chunk.content.startswith("<function="):
                    full_response_content += chunk.content
                    yield False, chunk.content

                # Hợp nhất kiểm tra tool call:
                # 1. Kiểm tra trường tool_calls chính thức
                # 2. Kiểm tra content có phải là tool call trá hình không (tools call trá hình là dạng <function=tool_name>, ví dụ: <function=search>)
                has_tool_calls_in_kwargs = (
                    chunk.additional_kwargs
                    and "tool_calls" in chunk.additional_kwargs
                    and chunk.additional_kwargs["tool_calls"]
                )
                is_content_a_tool_call = chunk.content.startswith("<function=")

                if (
                    has_tool_calls_in_kwargs or is_content_a_tool_call
                ) and not tool_call_detected:
                    # Nếu tool call nằm trong content, chúng ta cần tạo lại cấu trúc tool_call
                    if is_content_a_tool_call:
                        # Đây là một giả định đơn giản, cần điều chỉnh nếu định dạng phức tạp hơn
                        tool_name = chunk.content.split("=")[1].strip(">")
                        reconstructed_tool_call = {
                            "id": f"call_{tool_name}",
                            "function": {"name": tool_name, "arguments": "{}"},
                            "type": "function",
                        }
                        tool_calls.append(reconstructed_tool_call)
                    else:
                        tool_calls.extend(chunk.additional_kwargs["tool_calls"])

                    tool_call_detected = True

        # Phase 2: if tool call, execute tools and return messages
        if tool_calls:
            ai_msg = AIMessage(
                content=full_response_content,
                additional_kwargs={"tool_calls": tool_calls},
            )
            messages.append(ai_msg)

            messages = await self._execute_tools(
                tool_calls, messages, session_id, user_id
            )
            yield True, messages

    @semantic_cache_llms.cache(namespace="post-cache")
    async def _rag_generation(
        self,
        messages: list,
        question: str,
        chat_history: list[dict],
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        """Phase 3: RAG generation với streaming output"""
        self._update_trace_context(session_id, user_id)

        context_str = build_context(messages)
        logger.info(f"Generated Context String: '{context_str}'")
        # RAG prompt với context
        prompt = self.prompt_rag.get_langchain_prompt(
            chat_history="\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in chat_history
            ),
            question=question,
            context=context_str,
        )

        # Stream RAG response với tracing
        async for chunk in self.llm_with_tools.astream(
            prompt,
            {
                "callbacks": [self.langfuse_handler],
                "metadata": {
                    "langfuse_session_id": session_id,
                    "langfuse_user_id": user_id,
                },
            },
        ):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield content

    async def generate_stream(
        self,
        question: str,
        chat_history: list[dict] | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        """Generate streaming response with RAG integration"""
        try:
            if chat_history is None:
                chat_history = []

            is_tool_call = False
            messages = None

            # Create a span for the message creation part
            with self.langfuse.start_as_current_span(
                name="create_message_sse"
            ) as create_message_span:
                async for has_tools, data in self._create_message(
                    question, chat_history, session_id, user_id
                ):
                    if has_tools:
                        is_tool_call = True
                        messages = data
                    else:
                        yield data
                # Update the span with the result of this step
                create_message_span.update(output={"is_tool_call": is_tool_call})

            if is_tool_call:
                # Create a span for the RAG generation part
                with self.langfuse.start_as_current_span(
                    name="rag_generation_sse"
                ) as rag_span:
                    # We need to collect the response to update the span
                    full_rag_response = ""
                    async for chunk in self._rag_generation(
                        messages=messages,
                        question=question,
                        chat_history=chat_history,
                        session_id=session_id,
                        user_id=user_id,
                    ):
                        full_rag_response += chunk
                        yield chunk
                    # Update the span with the full response
                    rag_span.update(output=full_rag_response)

        except Exception as e:
            logger.error(f"Error in generate_stream(): {e}")
            raise
