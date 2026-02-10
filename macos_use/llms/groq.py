import os
import json
import logging
from typing import Iterator, AsyncIterator, List, Optional, Any, Union, overload
from groq import Groq, AsyncGroq
from pydantic import BaseModel
from macos_use.llms.base import BaseChatLLM
from macos_use.llms.views import ChatLLMResponse, ChatLLMUsage, Metadata
from macos_use.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ImageMessage, ToolMessage
from macos_use.tool import Tool

logger = logging.getLogger(__name__)

class ChatGroq(BaseChatLLM):
    """
    Groq LLM implementation following the BaseChatLLM protocol.
    
    Groq provides ultra-fast inference for open-source models like:
    - Llama 3.3 70B
    - Mixtral
    - Gemma 2
    """
    
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the Groq LLM.

        Args:
            model (str): The model name to use.
            api_key (str, optional): Groq API key. Defaults to GROQ_API_KEY environment variable.
            base_url (str, optional): Base URL for the API.
            timeout (float): Request timeout.
            max_retries (int): Maximum number of retries.
            temperature (float, optional): Sampling temperature.
            **kwargs: Additional arguments for chat completions.
        """
        self._model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.temperature = temperature
        
        self.client = Groq(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.aclient = AsyncGroq(
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
        return "groq"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """
        Convert BaseMessage objects to Groq-compatible message dictionaries.
        """
        groq_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                groq_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                groq_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, ImageMessage):
                # Groq has limited vision support for some models, but usually follow OpenAI format
                content_list = []
                if msg.content:
                    content_list.append({"type": "text", "text": msg.content})
                
                b64_imgs = msg.convert_images(format="base64")
                for b64 in b64_imgs:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{msg.mime_type};base64,{b64}"}
                    })
                groq_messages.append({"role": "user", "content": content_list})
            elif isinstance(msg, AIMessage):
                groq_messages.append({"role": "assistant", "content": msg.content or ""})
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
                groq_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                groq_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.id,
                    "content": msg.content or ""
                })
        return groq_messages

    def _convert_tools(self, tools: List[Tool]) -> List[dict]:
        """
        Convert Tool objects to Groq-compatible tool definitions.
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
        Process Groq API response into ChatLLMResponse object.
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
        groq_messages = self._convert_messages(messages)
        groq_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": groq_messages,
            **self.kwargs
        }
        
        # Only add tools if they exist
        if groq_tools:
            params["tools"] = groq_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        # Handle structured output and json_mode
        if structured_output:
            # Groq doesn't have beta.parse, so use json_mode and manual parsing
            params["response_format"] = {"type": "json_object"}
        elif json_mode:
            params["response_format"] = {"type": "json_object"}
            
        response = self.client.chat.completions.create(**params)
        
        if structured_output:
            try:
                # Parse the JSON response into the structured output
                content_text = response.choices[0].message.content
                if content_text:
                    parsed = structured_output.model_validate_json(content_text)
                else:
                    # Fallback to empty object if no content
                    parsed = structured_output()
                
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
        groq_messages = self._convert_messages(messages)
        groq_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": groq_messages,
            **self.kwargs
        }
        
        if groq_tools:
            params["tools"] = groq_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if structured_output:
            params["response_format"] = {"type": "json_object"}
        elif json_mode:
            params["response_format"] = {"type": "json_object"}
            
        response = await self.aclient.chat.completions.create(**params)
        
        if structured_output:
            try:
                content_text = response.choices[0].message.content
                if content_text:
                    parsed = structured_output.model_validate_json(content_text)
                else:
                    parsed = structured_output()
                
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
        groq_messages = self._convert_messages(messages)
        groq_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": groq_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self.kwargs
        }
        
        if groq_tools:
            params["tools"] = groq_tools
        
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
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
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
        groq_messages = self._convert_messages(messages)
        groq_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": groq_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self.kwargs
        }
        
        if groq_tools:
            params["tools"] = groq_tools
        
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
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
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
        # Context windows vary by model
        context_window = 128000  # Default
        
        if "llama-3.3" in self._model or "llama-3.1" in self._model:
            context_window = 128000
        elif "mixtral" in self._model:
            context_window = 32768
        elif "gemma-2" in self._model:
            context_window = 8192
        
        return Metadata(
            name=self._model,
            context_window=context_window,
            owned_by="groq"
        )