from fastapi import FastAPI, Request, HTTPException
import time
import requests
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import threading
import logging

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/models")
async def get_available_models():
    """Get list of all available Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            return {"models": models}
        else:
            raise HTTPException(status_code=500, detail="Failed to fetch models from Ollama")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama server is not running or unreachable")

def query_ollama_model(model, prompt, temperature, max_tokens):
    try:
        start = time.time()
        response = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        })
        end = time.time()
        latency_ms = (end - start) * 1000
        if response.status_code != 200:
            return {
                "model": model,
                "error": f"Ollama API error: {response.text}",
                "latency_ms": latency_ms
            }
        result = response.json()
        completion = result.get("response", "")
        prompt_eval_count = result.get("prompt_eval_count", 0)
        eval_count = result.get("eval_count", 0)
        return {
            "model": model,
            "response": completion,
            "usage": {
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count,
                "total_tokens": prompt_eval_count + eval_count
            },
            "latency_ms": latency_ms
        }
    except requests.exceptions.ConnectionError:
        return {
            "model": model,
            "error": "Ollama server is not running or unreachable",
            "latency_ms": None
        }

@app.post("/llm")
async def run_prompt(data: dict):
    prompt = data["prompt"]
    model = data.get("model", "llama3:latest")
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 200)
    compare_all = data.get("compare_all", False)

    # Get available models
    try:
        models_response = requests.get(f"{OLLAMA_URL}/api/tags")
        if models_response.status_code == 200:
            available_models = [m["name"] for m in models_response.json().get("models", [])]
        else:
            raise HTTPException(status_code=500, detail="Failed to fetch models from Ollama")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama server is not running or unreachable")

    if compare_all:
        results = []
        start = time.time()
        for m in available_models:
            result = query_ollama_model(m, prompt, temperature, max_tokens)
            results.append(result)
        end = time.time()
        return {
            "results": results,
            "latency_ms": (end - start) * 1000,
            "models_compared": available_models
        }

    # Single model (default)
    if model not in available_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{model}' not found. Available models: {available_models}"
        )

    start = time.time()
    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        })
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama API error: {response.text}")
        result = response.json()
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama server is not running or unreachable")
    end = time.time()

    completion = result.get("response", "")
    prompt_eval_count = result.get("prompt_eval_count", len(prompt.split()))
    eval_count = result.get("eval_count", len(completion.split()))

    return {
        "response": completion,
        "latency_ms": (end - start) * 1000,
        "usage": {
            "prompt_tokens": prompt_eval_count,
            "completion_tokens": eval_count,
            "total_tokens": prompt_eval_count + eval_count
        },
        "model": model
    }
