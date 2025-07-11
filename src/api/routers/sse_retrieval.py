from fastapi import APIRouter, Depends, status
from src.api.dependencies.rag import get_rag_service
from src.schemas.api.requests import UserInput
from src.services.application.rag import Rag
from fastapi.responses import StreamingResponse
import asyncio
import uuid
import json

router = APIRouter()


@router.post(
    "/",
    status_code=status.HTTP_200_OK,
)
async def retrieve_restaurants(
    input: UserInput,
    rag_service: Rag = Depends(get_rag_service),
):
    try:
        # Check và generate session_id/user_id ở router
        session_id = input.session_id or str(uuid.uuid4())
        user_id = input.user_id or f"user_{str(uuid.uuid4())[:8]}"
        
        async def generate_response():
            # Gửi metadata trước
            metadata = {
                "session_id": session_id,
                "user_id": user_id
            }
            yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"
            
            # Stream response
            async for chunk in rag_service.get_sse_response(
                question=input.user_input,
                session_id=session_id,
                user_id=user_id,
            ):
                yield chunk
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
        )
    except asyncio.TimeoutError:
        return StreamingResponse("event: responseUpdate\ndata: [Timeout reached.]")
