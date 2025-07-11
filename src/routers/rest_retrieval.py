from fastapi import APIRouter, Depends, status
from src.dependencies.rag import get_rag_service
from src.schemas.in_output import UserInput, ResponseOutput
from src.services.rag import Rag

router = APIRouter()


@router.post(
    "/",
    status_code=status.HTTP_200_OK,
    response_model=ResponseOutput,
)
async def retrieve_restaurants(
    input: UserInput,
    rag_service: Rag = Depends(get_rag_service),
):  
    print("You are in rest api")
    response = await rag_service.get_response(
        question=input.user_input,
        session_id=input.session_id,
        user_id=input.user_id,
    )

    return response
