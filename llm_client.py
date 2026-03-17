import httpx
import json
import logging
from typing import Dict, Any, Optional

from config import OLLAMA_API_URL, CODING_MODEL, TEMPERATURES, MAX_OUTPUT_TOKENS

async def run_ollama_api(
    prompt: str,
    phase_name: str,
    timeout_sec: int,
) -> Dict[str, Any]:
    """Execute a raw request against the local Ollama API."""
    # Build request payload with deterministic parameters
    payload = {
        "model": CODING_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURES.get(phase_name, 0.0),
            "num_predict": MAX_OUTPUT_TOKENS.get(phase_name, 1200),
            "seed": 42,  # Fixed seed for maximum determinism
        },
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        logging.error(f"Ollama API call failed: {exc}")
        return {"error": str(exc)}
