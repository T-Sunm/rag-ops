from pydantic import BaseModel, Field


class ResponseOutput(BaseModel):
    response: str = Field(
        description="AI response to user question"
    )
    session_id: str = Field(
        description="Session ID for conversation tracking"
    )
    user_id: str = Field(
        description="User ID for conversation tracking"
    )