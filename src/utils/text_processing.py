from typing import List
from langchain_core.messages import BaseMessage, ToolMessage


def build_context(messages: List[BaseMessage]) -> str:
    tool_chunks: []
    for m in messages:
        if isinstance(m, ToolMessage):
            tool_chunks.append(str(m.content))

    context_str = "\n\n--- Retrieved Documents ---\n\n".join(tool_chunks)
    return context_str
