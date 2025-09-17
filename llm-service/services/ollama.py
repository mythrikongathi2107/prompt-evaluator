import httpx

async def query_ollama_model(model: str, prompt: str) -> str:
    print(f"Querying Ollama model: {model} with prompt: {prompt}")
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        # response.raise_for_status()
        return response.json()
