import os
import json
import logging
from typing import Iterator, AsyncIterator, List, Optional, Any, overload
from cerebras.cloud.sdk import Cerebras, AsyncCerebras
from pydantic import BaseModel
from macos_use.llms.base import BaseChatLLM
from macos_use.llms.views import ChatLLMResponse, ChatLLMUsage, Metadata
from macos_use.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ImageMessage, ToolMessage
from macos_use.tool import Tool

logger = logging.getLogger(__name__)

class ChatCerebras(BaseChatLLM):
    """
    Cerebras LLM implementation following the BaseChatLLM protocol.
    """
    
    def __init__(
        self,
        model: str = "llama-3.3-70b",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the Cerebras LLM.

        Args:
            model (str): The model name to use.
            api_key (str, optional): Cerebras API key. Defaults to CEREBRAS_API_KEY environment variable.
            base_url (str, optional): Base URL for the API.
            timeout (float): Request timeout.
            max_retries (int): Maximum number of retries.
            temperature (float, optional): Sampling temperature.
            **kwargs: Additional arguments for chat completions.
        """
        self._model = model
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        self.temperature = temperature
        
        self.client = Cerebras(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.aclient = AsyncCerebras(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "cerebras"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """
        Convert BaseMessage objects to Cerebras-compatible message dictionaries.
        """
        cerebras_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                cerebras_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                cerebras_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, ImageMessage):
                # Cerebras might not support images yet in the same way, but following OpenAI format
                content_list = []
                if msg.content:
                    content_list.append({"type": "text", "text": msg.content})
                
                b64_imgs = msg.convert_images(format="base64")
                for b64 in b64_imgs:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{msg.mime_type};base64,{b64}"}
                    })
                cerebras_messages.append({"role": "user", "content": content_list})
            elif isinstance(msg, AIMessage):
                cerebras_messages.append({"role": "assistant", "content": msg.content or ""})
            elif isinstance(msg, ToolMessage):
                # Reconstruct for history consistency
                tool_call = {
                    "id": msg.id,
                    "type": "function",
                    "function": {
                        "name": msg.name,
                        "arguments": json.dumps(msg.params)
                    }
                }
                cerebras_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                cerebras_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.id,
                    "content": msg.content or ""
                })
        return cerebras_messages

    def _convert_tools(self, tools: List[Tool]) -> List[dict]:
        """
        Convert Tool objects to Cerebras-compatible tool definitions.
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
        Process Cerebras API response into ChatLLMResponse object.
        """
        choice = response.choices[0]
        message = choice.message
        usage_data = response.usage
        
        usage = ChatLLMUsage(
            prompt_tokens=usage_data.prompt_tokens,
            completion_tokens=usage_data.completion_tokens,
            total_tokens=usage_data.total_tokens
        )
        
        content = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_call = message.tool_calls[0]
            try:
                params = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                params = {}
                
            content = ToolMessage(
                id=tool_call.id,
                name=tool_call.function.name,
                params=params
            )
        else:
            content = AIMessage(content=message.content or "")
            
        return ChatLLMResponse(
            content=content,
            usage=usage
        )

    @overload
    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        ...

    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        cerebras_messages = self._convert_messages(messages)
        cerebras_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": cerebras_messages,
            **self.kwargs
        }
        
        # Only add tools if they exist
        if cerebras_tools:
            params["tools"] = cerebras_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if structured_output or json_mode:
            params["response_format"] = {"type": "json_object"}
            
        response = self.client.chat.completions.create(**params)
        
        if structured_output:
            try:
                parsed = structured_output.model_validate_json(response.choices[0].message.content)
                return ChatLLMResponse(
                    content=parsed,
                    usage=ChatLLMUsage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens
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
        cerebras_messages = self._convert_messages(messages)
        cerebras_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": cerebras_messages,
            **self.kwargs
        }
        
        if cerebras_tools:
            params["tools"] = cerebras_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if structured_output or json_mode:
            params["response_format"] = {"type": "json_object"}
            
        response = await self.aclient.chat.completions.create(**params)
        
        if structured_output:
            try:
                parsed = structured_output.model_validate_json(response.choices[0].message.content)
                return ChatLLMResponse(
                    content=parsed,
                    usage=ChatLLMUsage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens
                    )
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse structured output: {e}")
        
        return self._process_response(response)

    @overload
    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        ...

    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        cerebras_messages = self._convert_messages(messages)
        cerebras_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": cerebras_messages,
            "stream": True,
            **self.kwargs
        }
        
        if cerebras_tools:
            params["tools"] = cerebras_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if structured_output or json_mode:
            params["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**params)
        
        # Accumulators for streamed tool calls
        tool_call_id = None
        tool_call_name = None
        tool_call_args = ""
        
        for chunk in response:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            
            if delta.content:
                yield ChatLLMResponse(content=AIMessage(content=delta.content))
            
            # Accumulate tool call deltas
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                tc_delta = delta.tool_calls[0]
                if tc_delta.id:
                    tool_call_id = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_call_name = tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_call_args += tc_delta.function.arguments
        
        # Yield accumulated tool call as final response
        if tool_call_id and tool_call_name:
            try:
                tool_params = json.loads(tool_call_args)
            except json.JSONDecodeError:
                tool_params = {}
            yield ChatLLMResponse(
                content=ToolMessage(id=tool_call_id, name=tool_call_name, params=tool_params)
            )

    @overload
    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        ...

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        cerebras_messages = self._convert_messages(messages)
        cerebras_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": cerebras_messages,
            "stream": True,
            **self.kwargs
        }
        
        if cerebras_tools:
            params["tools"] = cerebras_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if structured_output or json_mode:
            params["response_format"] = {"type": "json_object"}
        
        response = await self.aclient.chat.completions.create(**params)
        
        # Accumulators for streamed tool calls
        tool_call_id = None
        tool_call_name = None
        tool_call_args = ""
        
        async for chunk in response:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            
            if delta.content:
                yield ChatLLMResponse(content=AIMessage(content=delta.content))
            
            # Accumulate tool call deltas
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                tc_delta = delta.tool_calls[0]
                if tc_delta.id:
                    tool_call_id = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_call_name = tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_call_args += tc_delta.function.arguments
        
        # Yield accumulated tool call as final response
        if tool_call_id and tool_call_name:
            try:
                tool_params = json.loads(tool_call_args)
            except json.JSONDecodeError:
                tool_params = {}
            yield ChatLLMResponse(
                content=ToolMessage(id=tool_call_id, name=tool_call_name, params=tool_params)
            )

    def get_metadata(self) -> Metadata:
        return Metadata(
            name=self._model,
            context_window=8192,  # Typical for llama3 on Cerebras, can be adjusted
            owned_by="cerebras"
        )
