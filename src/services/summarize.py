from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from langchain.tools import StructuredTool
from langfuse.langchain import CallbackHandler
from langfuse import get_client
from src.utils import logger
from langchain_openai import ChatOpenAI
from src.settings import SETTINGS

class SummarizeService:
    def __init__(self, langfuse_handler: CallbackHandler):
        self.langfuse = get_client()
        # self.prompt_summarize = self.langfuse.get_prompt(
        #     "summarize_service",
        #     label="production",
        #     type="chat",
        # )
        self.llm = ChatOpenAI(
            openai_api_base="http://127.0.0.1:1234/v1",
            temperature=SETTINGS.LLMs_TEMPERATURE,
            openai_api_key="dummy",
            streaming=True,
        )
    async def _summarize_and_truncate_history(self, chat_history: list[dict], max_length: int = 4) -> list[dict]:
        """Summary 4 messages cũ nhất và giữ lại phần còn lại"""
        if len(chat_history) <= max_length:
            return chat_history
        
        try:
            # Lấy 6 messages cũ nhất để summary
            old_messages = chat_history[:max_length]  # 6 messages đầu
            remaining_messages = chat_history[max_length:]  # Phần còn lại
            
            # Tạo summary từ 6 messages cũ
            old_conversation = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in old_messages
            ])
            
            summary_prompt = f"""Summarize this conversation in Vietnamese, keeping key information:
                {old_conversation}
                
            Summary (in 2-3 sentences):"""
            
            # Call LLM để summary
            summary_msg = await self.llm.ainvoke([
                {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                {"role": "user", "content": summary_prompt}
            ])
            
            summary_content = summary_msg.content if isinstance(summary_msg.content, str) else str(summary_msg.content)
            
            # Tạo history mới: summary + remaining messages
            summarized_history = [
                {"role": "system", "content": f"Previous conversation summary: {summary_content}"}
            ] + remaining_messages
            
            logger.info(f"Summarized {len(old_messages)} old messages, kept {len(remaining_messages)} recent messages")
            return summarized_history
            
        except Exception as e:
            logger.error(f"Error summarizing history: {e}")
            # Fallback: chỉ lấy recent messages
            return chat_history[-max_length:]