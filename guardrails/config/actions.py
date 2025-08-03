from typing import Optional
from nemoguardrails.actions import action
from src.services.application.rag import rag_service


generator_service = rag_service.generator_service


async def get_query_response(user_question, session_id, user_id):
    history = rag_service._get_session_history(session_id)
    print("length of history is ", len(history))
    return await generator_service.generate(user_question, history, session_id, user_id)


@action(is_system_action=True)
async def user_query(context: Optional[dict] = None):
    """Function to invoke the QA chain to query user message."""
    messages = context.get("user_message", [])

    user_question = None
    session_id = None
    user_id = None
    for message in messages:
        if message.get("role") == "user":
            user_question = message.get("content")
        elif message.get("role") == "context":
            context_content = message.get("content", {})
            session_id = context_content.get("session_id")
            user_id = context_content.get("user_id")

    if not user_question:
        return "Could not find user message in the context."

    return await get_query_response(user_question, session_id, user_id)
