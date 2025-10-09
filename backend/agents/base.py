"""
Base Agent Class for ReddyGo AI Agents

Common functionality and configuration for all agents.
"""

import os
from typing import Dict, List, Any, Optional, Callable
from openai import OpenAI


class BaseAgent:
    """
    Base class for all ReddyGo AI agents.

    Provides:
    - OpenAI client initialization
    - Common configuration
    - Logging and error handling
    - Tool execution framework
    """

    def __init__(
        self,
        name: str,
        instructions: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7
    ):
        """
        Initialize base agent.

        Args:
            name: Agent name
            instructions: System instructions for the agent
            model: OpenAI model to use
            temperature: Model temperature (0-1)
        """
        self.name = name
        self.instructions = instructions
        self.model = model
        self.temperature = temperature

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)

        # Tools registry (defined by subclasses)
        self.tools: List[Dict[str, Any]] = []

    def add_tool(self, func: Callable, description: str, parameters: Dict[str, Any]):
        """
        Register a tool function.

        Args:
            func: Function to call
            description: Tool description
            parameters: JSON Schema for parameters
        """
        tool_def = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": parameters
            }
        }

        self.tools.append(tool_def)

    async def run(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Run agent with a user message.

        Args:
            user_message: User input
            context: Optional context data
            stream: Whether to stream response

        Returns:
            Agent response
        """
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": user_message}
        ]

        # Add context if provided
        if context:
            context_str = f"Context: {context}"
            messages.append({"role": "system", "content": context_str})

        try:
            if stream:
                return await self._run_streaming(messages)
            else:
                return await self._run_sync(messages)

        except Exception as e:
            return {
                "error": str(e),
                "agent": self.name,
                "status": "failed"
            }

    async def _run_sync(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute agent synchronously."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools if self.tools else None,
            temperature=self.temperature
        )

        message = response.choices[0].message

        result = {
            "agent": self.name,
            "status": "success",
            "content": message.content,
            "tool_calls": []
        }

        # Handle tool calls
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
                for tc in message.tool_calls
            ]

        # Token usage
        result["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return result

    async def _run_streaming(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute agent with streaming."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools if self.tools else None,
            temperature=self.temperature,
            stream=True
        )

        content_chunks = []

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content_chunks.append(chunk.choices[0].delta.content)

        return {
            "agent": self.name,
            "status": "success",
            "content": "".join(content_chunks),
            "streaming": True
        }

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # GPT-4o-mini pricing
        costs = {
            "gpt-4o-mini": {
                "input": 0.003 / 1000,   # $0.003 per 1K tokens
                "output": 0.012 / 1000   # $0.012 per 1K tokens
            },
            "gpt-4": {
                "input": 0.03 / 1000,
                "output": 0.06 / 1000
            }
        }

        model_costs = costs.get(self.model, costs["gpt-4o-mini"])

        total_cost = (input_tokens * model_costs["input"]) + (output_tokens * model_costs["output"])

        return round(total_cost, 6)
