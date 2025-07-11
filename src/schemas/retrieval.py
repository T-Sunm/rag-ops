from pydantic import BaseModel, Field

class SearchArgs(BaseModel):
    question: str = Field(
        description="User input",
        default="What do beetles eat?",
    )
    top_k: int = Field(
        description="Number of results to return",
        default=3,
    )
    with_score: bool = Field(
        description="Whether to return the score of the results",
        default=False,
    )