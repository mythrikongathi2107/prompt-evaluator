import os

MODEL_CONFIGS = [
    {
        "name": "mistral",
        "provider": "ollama",
    },
    {
        "name": "phi3",
        "provider": "ollama",
    },
    {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "provider": "huggingface",
        "api_key": os.getenv("HUGGINGFACE_API_KEY")
    },
    {
        "name": "command-r-plus",
        "provider": "cohere",
        "api_key": os.getenv("COHERE_API_KEY")
    }
]
