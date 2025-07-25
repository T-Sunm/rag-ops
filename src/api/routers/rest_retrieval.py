from fastapi import APIRouter, Depends, status
from nemoguardrails import LLMRails
from src.api.dependencies.rag import get_rag_service
from src.api.dependencies.guarails import get_guardrails
from src.schemas.api.requests import UserInput
from src.schemas.api.response import ResponseOutput
from src.services.application.rag import Rag

router = APIRouter()


@router.post(
    "/",
    status_code=status.HTTP_200_OK,
    response_model=ResponseOutput,
)
async def retrieve_restaurants(
    input: UserInput,
    rag_service: Rag = Depends(get_rag_service),
    guardrails: LLMRails = Depends(get_guardrails),
):
    print("You are in rest api")
    response = await rag_service.get_response(
        question=input.user_input,
        session_id=input.session_id,
        user_id=input.user_id,
        guardrails=guardrails,
    )

    return response
