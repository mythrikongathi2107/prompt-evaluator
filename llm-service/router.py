from fastapi import APIRouter
from pydantic import BaseModel
from evaluator import evaluate_prompt

router = APIRouter()

class PromptRequest(BaseModel):
    prompt: str

@router.post("/compare")
async def compare_models(request: PromptRequest):
    return await evaluate_prompt(request.prompt)
