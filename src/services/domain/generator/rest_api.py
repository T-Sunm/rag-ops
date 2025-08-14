from langchain_core.messages import SystemMessage
from src.cache.semantic_cache import semantic_cache_llms
from src.utils.text_processing import build_context
from .base import BaseGeneratorService
from langfuse import observe
from src.utils import logger


class RestApiGeneratorService(BaseGeneratorService):
    """Generator service dành cho REST API"""

    @observe(name="initial_llm_call_rest_api")
    async def _initial_llm_call(
        self,
        question: str,
        chat_history: list[dict],
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        """Phase 1: Initial LLM call để kiểm tra tool calls"""
        self._update_trace_context(session_id, user_id)
        # Format chat history to be included in the prompt
        formatted_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in chat_history
        )
        prompt_template = self.prompt_userinput.get_langchain_prompt(
            question=question, chat_history=formatted_history
        )
        messages = [SystemMessage(content=prompt_template)]
        ai_msg = await self.llm_with_tools.ainvoke(
            messages,
            {
                "callbacks": [self.langfuse_handler],
                "metadata": {  # trace attributes
                    "langfuse_session_id": session_id,
                    "langfuse_user_id": user_id,
                },
            },
        )
        return ai_msg, messages

    @observe(name="create_message_rest_api")
    async def _create_message(
        self,
        question: str,
        chat_history: list[dict],
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        # Phase 1: Initial LLM call with chat history
        ai_msg, messages = await self._initial_llm_call(
            question, chat_history, session_id, user_id
        )

        messages.append(ai_msg)

        # Kiểm tra tool calls
        tool_calls = ai_msg.additional_kwargs.get("tool_calls", [])

        if not tool_calls:
            # Không có tool calls - trả về answer trực tiếp
            answer = self.clear_think.sub("", ai_msg.content).strip()
            return False, answer

        # Phase 2: Thực thi tools
        messages = await self._execute_tools(tool_calls, messages, session_id, user_id)

        return True, messages

    @observe(name="rag_generation_rest_api")
    @semantic_cache_llms.cache(namespace="post-cache")
    async def _rag_generation(
        self,
        messages: list,
        question: str,
        chat_history: list[dict],
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        """Phase 3: RAG generation với context từ tools"""
        self._update_trace_context(session_id, user_id)

        context_str = build_context(messages)

        # RAG prompt với context
        prompt = self.prompt_rag.get_langchain_prompt(
            chat_history="\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in chat_history
            ),
            question=question,
            context=context_str,
        )

        # Final LLM call - không cần callbacks vì đã có built-in
        raw = await self.llm_with_tools.ainvoke(
            prompt,
            {
                "callbacks": [self.langfuse_handler],
                "metadata": {
                    "langfuse_session_id": session_id,
                    "langfuse_user_id": user_id,
                },
            },
        )
        content = raw.content if isinstance(raw.content, str) else str(raw.content)
        answer = self.clear_think.sub("", content).strip()

        return answer

    @observe(name="generate_rest_api")
    async def generate_rest_api(
        self,
        question: str,
        chat_history: list[dict],
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        try:
            has_tools, result = await self._create_message(
                question, chat_history, session_id, user_id
            )

            if not has_tools:
                # Không có tools - trả về answer trực tiếp
                return result

            # Có tools - tiếp tục với RAG prompt
            messages = result
            answer = await self._rag_generation(
                messages=messages,
                question=question,
                chat_history=chat_history,
                session_id=session_id,
                user_id=user_id,
            )

            return answer

        except Exception as e:
            logger.error(f"Error in generate(): {e}")
            raise
