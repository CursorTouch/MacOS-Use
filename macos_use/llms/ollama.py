import os
import json
import uuid
import logging
from typing import Iterator, AsyncIterator, List, Optional, Any, overload
import ollama
from ollama import Client, AsyncClient
from pydantic import BaseModel
from macos_use.llms.base import BaseChatLLM
from macos_use.llms.views import ChatLLMResponse, ChatLLMUsage, Metadata
from macos_use.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ImageMessage, ToolMessage
from macos_use.tool import Tool

logger = logging.getLogger(__name__)

class ChatOllama(BaseChatLLM):
    """
    Ollama LLM implementation following the BaseChatLLM protocol.
    """
    
    def __init__(
        self,
        model: str = "llama3.1",
        host: Optional[str] = None,
        timeout: float = 600.0,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the Ollama LLM.

        Args:
            model (str): The model name to use.
            host (str, optional): Ollama host URL. Defaults to OLLAMA_HOST environment variable or localhost.
            timeout (float): Request timeout.
            temperature (float, optional): Sampling temperature.
            **kwargs: Additional arguments for chat.
        """
        self._model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.temperature = temperature
        
        self.client = Client(host=self.host, timeout=timeout)
        self.aclient = AsyncClient(host=self.host, timeout=timeout)
        self.kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "ollama"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """
        Convert BaseMessage objects to Ollama-compatible message dictionaries.
        """
        ollama_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                ollama_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                ollama_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, ImageMessage):
                b64_imgs = msg.convert_images(format="base64")
                ollama_messages.append({
                    "role": "user",
                    "content": msg.content or "",
                    "images": b64_imgs
                })
            elif isinstance(msg, AIMessage):
                ollama_messages.append({"role": "assistant", "content": msg.content or ""})
            elif isinstance(msg, ToolMessage):
                # Ollama expects assistant message with tool_calls followed by tool message
                # Reconstruct for history consistency
                ollama_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": msg.name,
                            "arguments": msg.params
                        }
                    }]
                })
                ollama_messages.append({
                    "role": "tool",
                    "content": msg.content or ""
                })
        return ollama_messages

    def _convert_tools(self, tools: List[Tool]) -> List[dict]:
        """
        Convert Tool objects to Ollama-compatible tool definitions.
        """
        return [
            {
                "type": "function",
                "function": tool.json_schema
            }
            for tool in tools
        ]

    def _process_response(self, response: Any) -> ChatLLMResponse:
        """
        Process Ollama API response into ChatLLMResponse object.
        """
        message = response.get("message", {})
        
        # Usage extraction
        usage = ChatLLMUsage(
            prompt_tokens=response.get("prompt_eval_count", 0),
            completion_tokens=response.get("eval_count", 0),
            total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
        )
        
        content = None
        tool_calls = message.get("tool_calls", [])
        
        if tool_calls:
            # Handle tool call
            tool_call = tool_calls[0]
            func = tool_call.get("function", {})
            content = ToolMessage(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=func.get("name"),
                params=func.get("arguments", {})
            )
        else:
            # Regular completion
            content = AIMessage(content=message.get("content", ""))
            
        return ChatLLMResponse(
            content=content,
            usage=usage
        )

    @overload
    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        ...

    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        ollama_messages = self._convert_messages(messages)
        ollama_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": ollama_messages,
            **self.kwargs
        }
        
        # Only add tools if they exist
        if ollama_tools:
            params["tools"] = ollama_tools
        
        if self.temperature is not None:
            if "options" not in params:
                params["options"] = {}
            params["options"]["temperature"] = self.temperature
        
        if json_mode or structured_output:
            params["format"] = "json" if not structured_output else structured_output.model_json_schema()

        response = self.client.chat(**params)
        
        if structured_output:
            try:
                parsed = structured_output.model_validate_json(response["message"]["content"])
                return ChatLLMResponse(
                    content=parsed,
                    usage=ChatLLMUsage(
                        prompt_tokens=response.get("prompt_eval_count", 0),
                        completion_tokens=response.get("eval_count", 0),
                        total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
                    )
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse structured output: {e}")
                # Fall through to normal response processing

        return self._process_response(response)

    @overload
    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        ...

    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        ollama_messages = self._convert_messages(messages)
        ollama_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": ollama_messages,
            **self.kwargs
        }
        
        if ollama_tools:
            params["tools"] = ollama_tools
        
        if self.temperature is not None:
            if "options" not in params:
                params["options"] = {}
            params["options"]["temperature"] = self.temperature
        
        if json_mode or structured_output:
            params["format"] = "json" if not structured_output else structured_output.model_json_schema()

        response = await self.aclient.chat(**params)
        
        if structured_output:
            try:
                parsed = structured_output.model_validate_json(response["message"]["content"])
                return ChatLLMResponse(
                    content=parsed,
                    usage=ChatLLMUsage(
                        prompt_tokens=response.get("prompt_eval_count", 0),
                        completion_tokens=response.get("eval_count", 0),
                        total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
                    )
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse structured output: {e}")

        return self._process_response(response)

    @overload
    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        ...

    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        ollama_messages = self._convert_messages(messages)
        ollama_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": True,
            **self.kwargs
        }
        
        if ollama_tools:
            params["tools"] = ollama_tools
        
        if self.temperature is not None:
            if "options" not in params:
                params["options"] = {}
            params["options"]["temperature"] = self.temperature
        
        if json_mode:
            params["format"] = "json"

        response = self.client.chat(**params)
        
        for chunk in response:
            message = chunk.get("message", {})
            if "content" in message and message["content"]:
                yield ChatLLMResponse(content=AIMessage(content=message["content"]))
            
            # Handle tool calls in stream
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                tc = tool_calls[0]
                func = tc.get("function", {})
                yield ChatLLMResponse(
                    content=ToolMessage(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name=func.get("name"),
                        params=func.get("arguments", {})
                    )
                )

    @overload
    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        ...

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        ollama_messages = self._convert_messages(messages)
        ollama_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": True,
            **self.kwargs
        }
        
        if ollama_tools:
            params["tools"] = ollama_tools
        
        if self.temperature is not None:
            if "options" not in params:
                params["options"] = {}
            params["options"]["temperature"] = self.temperature
        
        if json_mode:
            params["format"] = "json"

        response = await self.aclient.chat(**params)
        
        async for chunk in response:
            message = chunk.get("message", {})
            if "content" in message and message["content"]:
                yield ChatLLMResponse(content=AIMessage(content=message["content"]))
            
            # Handle tool calls in stream
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                tc = tool_calls[0]
                func = tc.get("function", {})
                yield ChatLLMResponse(
                    content=ToolMessage(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name=func.get("name"),
                        params=func.get("arguments", {})
                    )
                )

    def get_metadata(self) -> Metadata:
        return Metadata(
            name=self._model,
            context_window=32768, # Common default for llama3
            owned_by="ollama"
        )
