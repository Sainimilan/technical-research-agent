# rag_core/llm.py
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_response(prompt: str, model: str = "mistral", stream: bool = False):
    """
    Generate a response from the local Ollama LLM.
    Supports streaming and non-streaming modes.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": stream
            },
            timeout=120
        )
        response.raise_for_status()

        if stream:
            def token_generator():
                for line in response.iter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        token = data.get("response", "")
                        yield token
                        if data.get("done", False):
                            break
            return token_generator()

        return response.json().get("response", "No response from model.")

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "❌ Cannot connect to Ollama. Make sure Ollama is running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            "❌ Ollama request timed out. Try a smaller model or shorter context."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"❌ Ollama returned an error: {e}")


def list_available_models() -> list:
    """Fetch all locally available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m["name"] for m in models] if models else ["mistral"]
    except Exception:
        return ["mistral"]  # Fallback defaults