"""
ReddyGo - Ollama Client for Local LLaMA
Run LLaMA models locally without API costs

Learning Focus:
- Local LLM deployment
- Model quantization
- Cost optimization
- Prompt engineering for smaller models
"""

import requests
from typing import Dict, List, Optional
import json


class OllamaClient:
    """
    Client for Ollama server running locally

    Models available:
    - llama3.1:8b (8 billion params, ~4GB RAM, fast)
    - llama3.1:70b (70 billion params, ~40GB RAM, powerful)
    - codellama (specialized for code)
    - mistral (alternative to LLaMA)
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Args:
            base_url: Ollama server URL (default: localhost:11434)
        """
        self.base_url = base_url
        self.models_cache = {}

    def is_available(self) -> bool:
        """
        Check if Ollama server is running
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """
        List all available models
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except:
            return []

    def generate(self,
                 model: str,
                 prompt: str,
                 system: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 500,
                 stream: bool = False) -> Dict:
        """
        Generate text completion

        Args:
            model: Model name (e.g., "llama3.1:8b")
            prompt: User prompt
            system: System prompt (role definition)
            temperature: Randomness (0=deterministic, 1=creative)
            max_tokens: Maximum response length
            stream: Whether to stream response

        Returns:
            Dict with:
                - response: str
                - model: str
                - tokens_generated: int
                - duration_ms: int
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                return {
                    "response": data.get("response", ""),
                    "model": data.get("model", model),
                    "tokens_generated": data.get("eval_count", 0),
                    "duration_ms": data.get("total_duration", 0) // 1_000_000,  # Convert to ms
                    "cost": 0.0  # Local = free!
                }
            else:
                return {
                    "response": f"Error: {response.status_code} - {response.text}",
                    "model": model,
                    "tokens_generated": 0,
                    "duration_ms": 0,
                    "cost": 0.0
                }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "model": model,
                "tokens_generated": 0,
                "duration_ms": 0,
                "cost": 0.0
            }

    def chat(self,
             model: str,
             messages: List[Dict[str, str]],
             temperature: float = 0.7,
             max_tokens: int = 500) -> Dict:
        """
        Chat completion (like OpenAI's chat API)

        Args:
            model: Model name
            messages: List of {"role": "user/assistant/system", "content": "..."}
            temperature: Randomness
            max_tokens: Max response length

        Returns:
            Same format as generate()
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                message = data.get("message", {})
                return {
                    "response": message.get("content", ""),
                    "model": data.get("model", model),
                    "tokens_generated": data.get("eval_count", 0),
                    "duration_ms": data.get("total_duration", 0) // 1_000_000,
                    "cost": 0.0
                }
            else:
                return {
                    "response": f"Error: {response.status_code}",
                    "model": model,
                    "tokens_generated": 0,
                    "duration_ms": 0,
                    "cost": 0.0
                }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "model": model,
                "tokens_generated": 0,
                "duration_ms": 0,
                "cost": 0.0
            }


class CoachWithLLaMA:
    """
    AI Coach using local LLaMA via Ollama
    Cost: $0 (runs on your machine)
    """

    def __init__(self, model: str = "llama3.1:8b"):
        self.client = OllamaClient()
        self.model = model
        self.system_prompt = """You are an expert fitness coach specializing in running and endurance training.

Your role:
- Provide personalized workout advice
- Help users reach their fitness goals
- Prevent injuries through smart training
- Encourage and motivate

Guidelines:
- Be concise and actionable
- Use specific numbers when possible
- Always prioritize safety
- Adapt advice to user's experience level"""

    def ask(self, user_query: str, context: Optional[str] = None) -> Dict:
        """
        Ask the AI coach a question

        Args:
            user_query: User's question
            context: Additional context (past workouts, memories, etc.)

        Returns:
            Response dict with advice
        """
        if not self.client.is_available():
            return {
                "response": "Local LLaMA is not running. Start Ollama with: ollama serve",
                "model": "error",
                "cost": 0.0
            }

        # Build full prompt with context
        if context:
            full_prompt = f"Context: {context}\\n\\nUser question: {user_query}"
        else:
            full_prompt = user_query

        # Generate response
        result = self.client.generate(
            model=self.model,
            prompt=full_prompt,
            system=self.system_prompt,
            temperature=0.7,
            max_tokens=300
        )

        return result


# Example usage & installation guide
if __name__ == "__main__":
    """
    Example: Using local LLaMA for coaching

    Installation:
    1. Install Ollama:
       curl -fsSL https://ollama.com/install.sh | sh

    2. Download LLaMA 3.1 (8B model, ~4GB):
       ollama pull llama3.1:8b

    3. Start Ollama server:
       ollama serve

    4. Run this script
    """

    print("="*80)
    print("ReddyGo - Local LLaMA Coach Demo")
    print("="*80)

    client = OllamaClient()

    # Check if Ollama is running
    if not client.is_available():
        print("\\n❌ Ollama is not running!")
        print("\\nTo install and start Ollama:")
        print("  1. curl -fsSL https://ollama.com/install.sh | sh")
        print("  2. ollama pull llama3.1:8b")
        print("  3. ollama serve")
        exit(1)

    print("\\n✅ Ollama is running!")

    # List available models
    models = client.list_models()
    print(f"\\nAvailable models: {', '.join(models) if models else 'None'}")

    if "llama3.1:8b" not in models:
        print("\\n⚠️  LLaMA 3.1 not found. Downloading...")
        print("Run: ollama pull llama3.1:8b")
        exit(1)

    # Initialize coach
    coach = CoachWithLLaMA(model="llama3.1:8b")

    # Test questions
    questions = [
        {
            "query": "I want to improve my 5km time. Currently running 28 minutes. What should I do?",
            "context": "User runs 3 times per week, mostly easy pace. No injuries."
        },
        {
            "query": "My knee hurts after long runs. What exercises can help?",
            "context": "User runs 30-40km per week. Pain starts after 10km."
        },
        {
            "query": "Create a simple 4-week training plan to run my first 10km.",
            "context": "User is a beginner, completed Couch to 5K program last month."
        }
    ]

    for i, test in enumerate(questions, 1):
        print(f"\\n{'='*80}")
        print(f"Question {i}: {test['query']}")
        print(f"Context: {test['context']}")
        print(f"{'='*80}")

        result = coach.ask(test['query'], context=test['context'])

        print(f"\\n{result['response']}")
        print(f"\\n📊 Stats:")
        print(f"  Model: {result['model']}")
        print(f"  Tokens: {result['tokens_generated']}")
        print(f"  Time: {result['duration_ms']}ms")
        print(f"  Cost: ${result['cost']:.4f} (FREE! 🎉)")

    print(f"\\n{'='*80}")
    print("Cost Comparison:")
    print("  LLaMA (local): $0.00")
    print("  GPT-4o-mini: ~$0.001 per query")
    print("  GPT-4: ~$0.02 per query")
    print(f"{'='*80}")
