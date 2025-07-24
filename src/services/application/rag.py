from src.cache.standard_cache import standard_cache
from src.services.domain.generator import GeneratorService
from src.services.domain.summarize import SummarizeService
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from src.config.settings import SETTINGS
from src.infrastructure.vector_stores.chroma_client import ChromaClientService
from src.schemas.domain.retrieval import SearchArgs
from src.utils import logger
from langfuse import observe
from langfuse.langchain import CallbackHandler
from langfuse import get_client
import uuid
from nemoguardrails import LLMRails
import asyncio, contextlib

import os
os.environ["LANGFUSE_PUBLIC_KEY"] = SETTINGS.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = SETTINGS.LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"]       = SETTINGS.LANGFUSE_HOST

class Rag:
    def __init__(self):
        self.llm = ChatOpenAI(
            **SETTINGS.llm_config
        )
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
            args_schema=SearchArgs
        )

        # Define tools dictionary
        self.tools = {
            "search_docs": self.search_tool
        }
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(list(self.tools.values())).with_config({
            "callbacks": [self.langfuse_handler]
        })

        # Initialize services
        self.generator_service = GeneratorService(
            llm_with_tools=self.llm_with_tools,
            tools=self.tools,
            langfuse_handler=self.langfuse_handler
        )
        
        self.summarize_service = SummarizeService(
            langfuse_handler=self.langfuse_handler
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
        self.session_histories[session_id].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ])

    async def _get_response_inner(self, question: str, chat_history: list, session_id: str, user_id: str, guardrails: LLMRails):
        """Core logic for getting response, wrapped by a timeout."""
        rag_task = asyncio.create_task(
            self.generator_service.generate(
                question=question,
                chat_history=list(chat_history),
                session_id=session_id,
                user_id=user_id,
            ),
            name="rag_generate",
        )

        # Format the prompt as a chat message list for the guardrails model
        chat_prompt = [{"role": "user", "content": question}]
        
        guardrail_task = asyncio.create_task(
            guardrails.generate_async(prompt=chat_prompt),
            name="guardrail_check",
        )

        # Wait for the first guardrail check on the user input
        guardrail_result = await guardrail_task
        print(guardrail_result)
        # If the result is different from the original question, the guardrail has intervened.
        if "sorry" in str(guardrail_result):
            rag_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await rag_task
            
            return {"response": guardrail_result, "session_id": session_id, "user_id": user_id}

        # If the first check passes, wait for the RAG process
        rag_output = await rag_task

        # Second guardrail check on the RAG output
        chat_prompt_output = [{"role": "user", "content": rag_output}]
        guardrail_task = asyncio.create_task(guardrails.generate_async(prompt=chat_prompt_output))
        guardrail_result = await guardrail_task

        # If the result is different, the guardrail blocked the output.
        if "sorry" in str(guardrail_result):
            return {"response": guardrail_result, "session_id": session_id, "user_id": user_id}
        
        return rag_output

    @observe(name="rag-service")
    @standard_cache.cache(ttl=300)
    async def get_response(self, question: str, session_id: str | None = None, user_id: str | None = None, guardrails: LLMRails | None = None):
        # ---------- ID Normalization ----------
        session_id = session_id or str(uuid.uuid4())
        user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"

        # ---------- History Retrieval & Summarization ----------
        chat_history = self._get_session_history(session_id)
        if len(chat_history) >= 6:
            chat_history = await self.summarize_service._summarize_and_truncate_history(
                chat_history, keep_last=4
            )
            self.session_histories[session_id] = chat_history

        # ---------- Concurrent Execution with Timeout ----------
        try:
            result = await asyncio.wait_for(
                self._get_response_inner(question, chat_history, session_id, user_id, guardrails),
                timeout=60.0
            )

            if isinstance(result, dict):
                return result
            
            rag_output = result

        except asyncio.TimeoutError:
            logger.error("Request timed out for session %s", session_id)
            timeout_msg = "Sorry, the request timed out."
            self._save_to_session_history(session_id, question, timeout_msg)
            return {"response": timeout_msg, "session_id": session_id, "user_id": user_id}
            
        # ---------- Save & Return ----------
        self._save_to_session_history(session_id, question, rag_output)
        logger.info(
            "Session %s | User %s | History len: %d",
            session_id, user_id, len(self.session_histories[session_id])
        )

        return {
            "response": rag_output,
            "session_id": session_id,
            "user_id": user_id,
        }

    async def get_sse_response(self, question: str, session_id: str, user_id: str):   
        # Lấy history từ memory
        chat_history = self._get_session_history(session_id)
        if len(chat_history) >= 6:
            chat_history = await self.summarize_service._summarize_and_truncate_history(
                chat_history,
                4
            )
            self.session_histories[session_id] = chat_history
            
        # Collect full response để save sau
        full_response = ""
        async for message in self.generator_service.generate_stream(
            question=question,
            chat_history=chat_history.copy(),
            session_id=session_id,
            user_id=user_id
        ):
            full_response += message
            yield f"event: responseUpdate\ndata: {message}\n\n"
        
        # Save conversation sau khi stream xong
        self._save_to_session_history(session_id, question, full_response)
        
        # Kết thúc stream
        yield "event: responseUpdate\ndata: [DONE]\n\n"
