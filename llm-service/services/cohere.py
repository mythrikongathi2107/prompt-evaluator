import httpx

async def query_cohere_model(api_key: str, prompt: str) -> str:
    url = "https://api.cohere.ai/v1/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "message": prompt,
        "model": "command-r-plus",
        "temperature": 0.5,
        "chat_history": []
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("text", "")
