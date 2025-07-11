from src.constants.prompt import temp_userinput, temp_rag
from langchain_core.messages import AIMessage
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
from langfuse import observe
class GeneratorService:
    def __init__(self, llm_with_tools: Runnable[LanguageModelInput, BaseMessage], tools: dict[str, StructuredTool], langfuse_handler: CallbackHandler):
        self.llm_with_tools = llm_with_tools
        self.tools = tools
        self.langfuse = get_client()
        self.prompt_userinput = self.langfuse.get_prompt(
            "userinput_service",
            label="production",
            type="chat",
        )
        self.prompt_rag = self.langfuse.get_prompt(
            "rag_service",
            label="production",
            type="chat",
        )
        self.clear_think = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
        self.langfuse_handler = langfuse_handler
    
    def _update_trace_context(self, session_id: str | None = None, user_id: str | None = None):
        """Helper method để update trace context và handler"""
        if session_id:
            self.langfuse.update_current_trace(session_id=session_id)
            self.langfuse_handler.session_id = session_id
        if user_id:
            self.langfuse.update_current_trace(user_id=user_id)
            self.langfuse_handler.user_id = user_id
    
    @observe(name="initial_llm_call")
    async def _initial_llm_call(self, question: str, session_id: str | None = None, user_id: str | None = None):
        """Phase 1: Initial LLM call để kiểm tra tool calls"""
        self._update_trace_context(session_id, user_id)
        messages = self.prompt_userinput.get_langchain_prompt(question=question)
        ai_msg = await self.llm_with_tools.ainvoke(messages)
        return ai_msg, messages
    
    async def _execute_tools(self, tool_calls: list, messages: list, session_id: str | None = None, user_id: str | None = None):
        self._update_trace_context(session_id, user_id)

        executed_tools = []
        for tool_call in tool_calls:
            name = tool_call["function"]["name"].lower()
            if name not in self.tools:
                raise ValueError(f"Unknown tool: {name}")
            
            tool_inst = self.tools[name]
            payload = json.loads(tool_call["function"]["arguments"])
            
            with self.langfuse.start_as_current_span(
                name=f"tool_{name}",
                input=payload,
                metadata={"tool_name": name}
            ) as span:
                
                if "tool_calls" in payload:
                    for call_args in payload["tool_calls"]:
                        # Trace từng call args nếu có nhiều
                        with self.langfuse.start_as_current_span(
                            name=f"tool_{name}_call",
                            input=call_args
                        ) as sub_span:
                            output = tool_inst.invoke(call_args)
                            sub_span.update(output=output)
                            
                        messages.append(ToolMessage(content=output, tool_call_id=tool_call.get("id")))
                        executed_tools.append({"tool_name": name, "output": output})
                else:
                    output = tool_inst.invoke(payload)
                    span.update(output=output)
                    messages.append(ToolMessage(content=output, tool_call_id=tool_call.get("id")))
                    executed_tools.append({"tool_name": name, "output": output})
        
        return messages, executed_tools
    
    @observe(name="rag_generation")
    async def _rag_generation(self, messages: list, question: str, chat_history: list[dict], 
                             session_id: str | None = None, user_id: str | None = None):
        """Phase 3: RAG generation với context từ tools"""
        self._update_trace_context(session_id, user_id)
        
        # Tạo context từ tool results
        tool_results = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                tool_results.append(content)
        context_str = "\n\n--- Retrieved Documents ---\n\n".join(tool_results)
        
        # RAG prompt với context
        prompt = self.prompt_rag.get_langchain_prompt(
            chat_history="\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in chat_history),
            question=question,
            context=context_str
        )
        
        # Final LLM call - không cần callbacks vì đã có built-in
        raw = await self.llm_with_tools.ainvoke(prompt)
        content = raw.content if isinstance(raw.content, str) else str(raw.content)
        answer = self.clear_think.sub("", content).strip()
        
        return answer
    
    @observe(name="create_message")
    async def _create_message(self, question: str, chat_history: list[dict], 
                             session_id: str | None = None, user_id: str | None = None):
        # Phase 1: Initial LLM call
        ai_msg, messages = await self._initial_llm_call(question, session_id, user_id)
        
        # Thêm AI response vào messages để duy trì conversation flow
        messages.append(ai_msg)
        
        # Thêm vào chat history
        chat_history.append({
            "role": "assistant", 
            "content": ai_msg.content, 
            "tool_calls": ai_msg.additional_kwargs.get("tool_calls", [])
        })
        
        # Kiểm tra tool calls
        tool_calls = ai_msg.additional_kwargs.get("tool_calls", [])
        
        if not tool_calls:
            # Không có tool calls - trả về answer trực tiếp
            content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
            answer = self.clear_think.sub("", content).strip()
            return False, answer, chat_history
        
        # Phase 2: Thực thi tools
        messages, executed_tools = await self._execute_tools(tool_calls, messages, session_id, user_id)
        
        # Thêm tool results vào chat history
        for tool_info in executed_tools:
            chat_history.append({
                "role": "tool", 
                "content": tool_info["output"], 
                "tool_name": tool_info["tool_name"]
            })
        
        return True, messages, chat_history
    
    @observe(name="generate")
    async def generate(self, question: str, chat_history: list[dict], 
                      session_id: str | None = None, user_id: str | None = None):
        try:
            has_tools, result, updated_chat_history = await self._create_message(
                question, chat_history, session_id, user_id
            )
            
            if not has_tools:
                # Không có tools - trả về answer trực tiếp
                return result
            
            # Có tools - tiếp tục với RAG prompt
            messages = result  # type: ignore  # result is guaranteed to be list when has_tools is True
            answer = await self._rag_generation(messages, question, updated_chat_history, session_id, user_id)
            
            return answer

        except Exception as e:
            logger.error(f"Error in generate(): {e}")
            raise

    @observe(name="generate_stream")
    async def generate_stream(self, question: str, chat_history: list[dict] | None = None,
                             session_id: str | None = None, user_id: str | None = None):
        try:
            if chat_history is None:
                chat_history = []
                
            has_tools, result, updated_chat_history = await self._create_message(
                question, chat_history, session_id, user_id
            )
            
            if not has_tools:
                yield result
                return
            
            # Có tools - stream final response với observability
            self._update_trace_context(session_id, user_id)
            messages = result
            full_response = ""
            think_tag_passed = False
            
            # Không cần callbacks vì đã có built-in
            async for chunk in self.llm_with_tools.astream(messages):
                if chunk.content:
                    content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    
                    if not think_tag_passed:
                        full_response += content
                        if "</think>" in full_response:
                            think_tag_passed = True
                            # Yield phần sau </think>
                            remaining = full_response.split("</think>", 1)[1]
                            if remaining:
                                yield remaining
                    else:
                        # Đã qua think tag, stream trực tiếp
                        yield content

        except Exception as e:
            logger.error(f"Error in generate_stream(): {e}")
            raise