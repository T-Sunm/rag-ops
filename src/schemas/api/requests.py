from pydantic import BaseModel, Field


class UserInput(BaseModel):
    user_input: str = Field(
        description="User input",
        default="What do beetles eat?",
    )
    session_id: str = Field(
        description="Session ID",
        default="1",
    )
    user_id: str = Field(
        description="User ID",
        default="1",
    )