import os
import json
import logging
from typing import Iterator, AsyncIterator, List, Optional, Any, Union, overload
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from macos_use.llms.base import BaseChatLLM
from macos_use.llms.views import ChatLLMResponse, ChatLLMUsage, Metadata
from macos_use.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ImageMessage, ToolMessage
from macos_use.tool import Tool

logger = logging.getLogger(__name__)

class ChatOpenRouter(BaseChatLLM):
    """
    OpenRouter LLM implementation following the BaseChatLLM protocol.
    Uses OpenAI SDK with OpenRouter base URL.
    
    OpenRouter provides access to multiple LLM providers through a unified API.
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 600.0,
        max_retries: int = 2,
        temperature: Optional[float] = None,
        default_headers: Optional[dict] = None,
        **kwargs
    ):
        """
        Initialize the OpenRouter LLM.

        Args:
            model (str): The model name (e.g., 'anthropic/claude-3-sonnet').
            api_key (str, optional): OpenRouter API key. Defaults to OPENROUTER_API_KEY.
            base_url (str): OpenRouter base URL.
            timeout (float): Request timeout.
            max_retries (int): Maximum retries.
            temperature (float, optional): Sampling temperature..
            **kwargs: Additional arguments for chat completions.
        """
        self._model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.temperature = temperature
        
        # Build headers
        headers = default_headers or {}
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=headers
        )
        self.aclient = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=headers
        )
        self.kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "open_router"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """
        Convert BaseMessage objects to OpenRouter-compatible message dictionaries.
        """
        openai_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, ImageMessage):
                content_list = []
                if msg.content:
                    content_list.append({"type": "text", "text": msg.content})
                
                b64_imgs = msg.convert_images(format="base64")
                for b64 in b64_imgs:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{msg.mime_type};base64,{b64}"}
                    })
                openai_messages.append({"role": "user", "content": content_list})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content or ""})
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
                openai_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.id,
                    "content": msg.content or ""
                })
        return openai_messages

    def _convert_tools(self, tools: List[Tool]) -> List[dict]:
        """
        Convert Tool objects to OpenRouter-compatible tool definitions.
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
        Process OpenRouter API response into ChatLLMResponse object.
        """
        choice = response.choices[0]
        message = choice.message
        usage_data = response.usage
        
        # Safely extract usage data
        usage = ChatLLMUsage(
            prompt_tokens=getattr(usage_data, 'prompt_tokens', 0),
            completion_tokens=getattr(usage_data, 'completion_tokens', 0),
            total_tokens=getattr(usage_data, 'total_tokens', 0)
        )
        
        content = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_call = message.tool_calls[0]
            try:
                params = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool arguments: {tool_call.function.arguments}")
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
        openai_messages = self._convert_messages(messages)
        openai_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": openai_messages,
            **self.kwargs
        }
        
        # Only add tools if they exist
        if openai_tools:
            params["tools"] = openai_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        
        # Note: OpenRouter doesn't support structured_output via beta.parse
        if structured_output:
            logger.warning("OpenRouter does not support structured_output via beta.parse. Using json_mode instead.")
            params["response_format"] = {"type": "json_object"}
            
        response = self.client.chat.completions.create(**params)
        
        if structured_output:
            # Try to parse the JSON response into the structured output
            try:
                content_text = response.choices[0].message.content
                parsed_data = json.loads(content_text)
                parsed_obj = structured_output(**parsed_data)
                
                usage = ChatLLMUsage(
                    prompt_tokens=getattr(response.usage, 'prompt_tokens', 0),
                    completion_tokens=getattr(response.usage, 'completion_tokens', 0),
                    total_tokens=getattr(response.usage, 'total_tokens', 0)
                )
                
                return ChatLLMResponse(
                    content=parsed_obj,
                    usage=usage
                )
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.error(f"Failed to parse structured output: {e}")
                # Fall through to normal response processing
        
        return self._process_response(response)

    @overload
    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        ...

    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        openai_messages = self._convert_messages(messages)
        openai_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": openai_messages,
            **self.kwargs
        }
        
        if openai_tools:
            params["tools"] = openai_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        
        if structured_output:
            logger.warning("OpenRouter does not support structured_output via beta.parse. Using json_mode instead.")
            params["response_format"] = {"type": "json_object"}
            
        response = await self.aclient.chat.completions.create(**params)
        
        if structured_output:
            try:
                content_text = response.choices[0].message.content
                parsed_data = json.loads(content_text)
                parsed_obj = structured_output(**parsed_data)
                
                usage = ChatLLMUsage(
                    prompt_tokens=getattr(response.usage, 'prompt_tokens', 0),
                    completion_tokens=getattr(response.usage, 'completion_tokens', 0),
                    total_tokens=getattr(response.usage, 'total_tokens', 0)
                )
                
                return ChatLLMResponse(
                    content=parsed_obj,
                    usage=usage
                )
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.error(f"Failed to parse structured output: {e}")
        
        return self._process_response(response)

    @overload
    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        ...

    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        openai_messages = self._convert_messages(messages)
        openai_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": openai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self.kwargs
        }
        
        if openai_tools:
            params["tools"] = openai_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**params)
        
        # Accumulators for streamed tool calls
        tool_call_id = None
        tool_call_name = None
        tool_call_args = ""
        final_usage = None
        
        for chunk in response:
            if not chunk.choices:
                if hasattr(chunk, 'usage') and chunk.usage:
                    final_usage = ChatLLMUsage(
                        prompt_tokens=getattr(chunk.usage, 'prompt_tokens', 0),
                        completion_tokens=getattr(chunk.usage, 'completion_tokens', 0),
                        total_tokens=getattr(chunk.usage, 'total_tokens', 0),
                    )
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
                content=ToolMessage(id=tool_call_id, name=tool_call_name, params=tool_params),
                usage=final_usage
            )
        elif final_usage:
            yield ChatLLMResponse(usage=final_usage)

    @overload
    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        ...

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        openai_messages = self._convert_messages(messages)
        openai_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": openai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self.kwargs
        }
        
        if openai_tools:
            params["tools"] = openai_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        
        response = await self.aclient.chat.completions.create(**params)
        
        # Accumulators for streamed tool calls
        tool_call_id = None
        tool_call_name = None
        tool_call_args = ""
        final_usage = None
        
        async for chunk in response:
            if not chunk.choices:
                if hasattr(chunk, 'usage') and chunk.usage:
                    final_usage = ChatLLMUsage(
                        prompt_tokens=getattr(chunk.usage, 'prompt_tokens', 0),
                        completion_tokens=getattr(chunk.usage, 'completion_tokens', 0),
                        total_tokens=getattr(chunk.usage, 'total_tokens', 0),
                    )
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
                content=ToolMessage(id=tool_call_id, name=tool_call_name, params=tool_params),
                usage=final_usage
            )
        elif final_usage:
            yield ChatLLMResponse(usage=final_usage)

    def get_metadata(self) -> Metadata:
        return Metadata(
            name=self._model,
            context_window=128000,  # Varies by model, this is a safe default
            owned_by="open_router"
        )