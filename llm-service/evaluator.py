import asyncio
from config import MODEL_CONFIGS
from services.ollama import query_ollama_model
from services.huggingface import query_hf_model
from services.cohere import query_cohere_model

async def call_model(model_config, prompt: str):
    provider = model_config["provider"]
    try:
        if provider == "ollama":
            return {
                "model": model_config["name"],
                "response": await query_ollama_model(model_config["name"], prompt)
            }
        elif provider == "huggingface":
            return {
                "model": model_config["name"],
                "response": await query_hf_model(model_config["name"], model_config["api_key"], prompt)
            }
        elif provider == "cohere":
            return {
                "model": model_config["name"],
                "response": await query_cohere_model(model_config["api_key"], prompt)
            }
        else:
            return {"model": model_config["name"], "response": "Unsupported provider"}
    except Exception as e:
        return {"model": model_config["name"], "response": f"Error: {str(e)}"}

async def evaluate_prompt(prompt: str):
    tasks = [call_model(model, prompt) for model in MODEL_CONFIGS]
    results = await asyncio.gather(*tasks)
    return {"prompt": prompt, "results": results}
