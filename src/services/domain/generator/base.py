from langchain.tools import StructuredTool
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import ToolMessage
from src.utils import logger
import json
import re
from langfuse.langchain import CallbackHandler
from langfuse import get_client
from abc import ABC, abstractmethod


class BaseGeneratorService(ABC):
    """Base class for REST API and SSE"""

    def __init__(
        self,
        llm_with_tools: Runnable[LanguageModelInput, BaseMessage],
        tools: dict[str, StructuredTool],
        langfuse_handler: CallbackHandler,
    ):
        self.llm_with_tools = llm_with_tools
        self.tools = tools
        self.langfuse = get_client()
        self.prompt_userinput = self.langfuse.get_prompt(
            "userinput_service",
            label="production",
            type="text",
        )
        self.prompt_rag = self.langfuse.get_prompt(
            "rag_service",
            label="production",
            type="text",
        )
        self.clear_think = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
        self.langfuse_handler = langfuse_handler

    def _update_trace_context(
        self, session_id: str | None = None, user_id: str | None = None
    ):
        """Helper method để update trace context và handler"""
        if session_id:
            self.langfuse.update_current_trace(session_id=session_id)
        if user_id:
            self.langfuse.update_current_trace(user_id=user_id)

    @abstractmethod
    async def _initial_llm_call(
        self,
        question: str,
        chat_history: list[dict],
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        pass

    @abstractmethod
    async def _create_message(
        self,
        question: str,
        chat_history: list[dict],
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        pass

    async def _execute_tools(
        self,
        tool_calls: list,
        messages: list,
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        self._update_trace_context(session_id, user_id)

        executed_tools = []
        for tool_call in tool_calls:
            name = tool_call["function"]["name"].lower()
            if name not in self.tools:
                raise ValueError(f"Unknown tool: {name}")

            tool_inst = self.tools[name]
            payload = json.loads(tool_call["function"]["arguments"])

            with self.langfuse.start_as_current_span(
                name=f"tool_{name}", input=payload, metadata={"tool_name": name}
            ) as span:

                if "tool_calls" in payload:
                    for call_args in payload["tool_calls"]:
                        # Trace từng call args nếu có nhiều
                        with self.langfuse.start_as_current_span(
                            name=f"tool_{name}_call", input=call_args
                        ) as sub_span:
                            output = tool_inst.invoke(call_args)
                            sub_span.update(output=output)

                        messages.append(
                            ToolMessage(
                                content=output, tool_call_id=tool_call.get("id")
                            )
                        )
                else:
                    output = tool_inst.invoke(payload)
                    span.update(output=output)
                    messages.append(
                        ToolMessage(content=output, tool_call_id=tool_call.get("id"))
                    )

        return messages

    @abstractmethod
    async def _rag_generation(
        self,
        messages: list,
        question: str,
        chat_history: list[dict],
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        pass
