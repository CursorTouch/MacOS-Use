import os
import json
import logging
from typing import Iterator, AsyncIterator, List, Optional, Any, Union, overload
from anthropic import Anthropic, AsyncAnthropic
from pydantic import BaseModel
from macos_use.llms.base import BaseChatLLM
from macos_use.llms.views import ChatLLMResponse, ChatLLMUsage, Metadata
from macos_use.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ImageMessage, ToolMessage
from macos_use.tool import Tool

logger = logging.getLogger(__name__)

class ChatAnthropic(BaseChatLLM):
    """
    Anthropic LLM implementation following the BaseChatLLM protocol.
    """
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        temperature: Optional[float] = None,
        enable_prompt_caching: bool = True,
        cache_system_prompt: bool = True,
        cache_tools: bool = True,
        cache_recent_messages: int = 0,  # Number of recent messages to cache
        **kwargs
    ):
        """
        Initialize the Anthropic LLM with prompt caching support.

        Args:
            model (str): The model name to use.
            api_key (str, optional): Anthropic API key.
            base_url (str, optional): Base URL for the API.
            timeout (float): Request timeout in seconds.
            max_retries (int): Maximum number of retries.
            temperature (float, optional): Sampling temperature.
            enable_prompt_caching (bool): Enable prompt caching feature.
            cache_system_prompt (bool): Add cache control to system prompt.
            cache_tools (bool): Add cache control to tools.
            cache_recent_messages (int): Number of recent messages to mark for caching.
            **kwargs: Additional arguments for messages.create.
        """
        self._model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        self.temperature = temperature
        self.enable_prompt_caching = enable_prompt_caching
        self.cache_system_prompt = cache_system_prompt
        self.cache_tools = cache_tools
        self.cache_recent_messages = cache_recent_messages
        
        self.client = Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.aclient = AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "anthropic"

    def _convert_messages(self, messages: List[BaseMessage]) -> tuple[Optional[Union[str, List[dict]]], List[dict]]:
        """
        Convert BaseMessage objects to Anthropic-compatible message dictionaries.
        Returns (system_prompt, messages).
        
        Adds cache control breakpoints based on configuration.
        """
        anthropic_messages = []
        system_prompt = None
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, HumanMessage):
                anthropic_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, ImageMessage):
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                
                b64_imgs = msg.convert_images(format="base64")
                for b64 in b64_imgs:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": msg.mime_type,
                            "data": b64
                        }
                    })
                anthropic_messages.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage):
                content = []
                if msg.thinking:
                    content.append({
                        "type": "thinking", 
                        "thinking": msg.thinking, 
                        "signature": msg.thinking_signature
                    })
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                anthropic_messages.append({"role": "assistant", "content": content if content else ""})
            elif isinstance(msg, ToolMessage):
                # Anthropic requires a tool_use block followed by a tool_result block
                if anthropic_messages and anthropic_messages[-1]["role"] == "assistant":
                    last_content = anthropic_messages[-1]["content"]
                    if isinstance(last_content, str):
                        last_content = [{"type": "text", "text": last_content}]
                    
                    tool_use_block = {
                        "type": "tool_use",
                        "id": msg.id,
                        "name": msg.name,
                        "input": msg.params
                    }
                    if isinstance(last_content, list):
                        last_content.append(tool_use_block)
                    anthropic_messages[-1]["content"] = last_content
                else:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": [{
                            "type": "tool_use",
                            "id": msg.id,
                            "name": msg.name,
                            "input": msg.params
                        }]
                    })
                
                # Add the tool result
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.id,
                        "content": msg.content or ""
                    }]
                })
        
        # Apply prompt caching to system prompt
        if system_prompt and self.enable_prompt_caching and self.cache_system_prompt:
            system_prompt = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        
        # Apply prompt caching to recent messages
        if self.enable_prompt_caching and self.cache_recent_messages > 0 and anthropic_messages:
            # Cache the last N user messages (caching works best on user turns)
            user_message_indices = [
                i for i, msg in enumerate(anthropic_messages) 
                if msg["role"] == "user"
            ]
            
            # Get the indices to cache (last N user messages)
            indices_to_cache = user_message_indices[-self.cache_recent_messages:]
            
            for idx in indices_to_cache:
                msg = anthropic_messages[idx]
                content = msg["content"]
                
                # Convert string content to list format for cache control
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                
                # Add cache control to the last content block
                if isinstance(content, list) and len(content) > 0:
                    # Only add cache control to the last block in the message
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                    msg["content"] = content
        
        return system_prompt, anthropic_messages

    def _convert_tools(self, tools: List[Tool]) -> List[dict]:
        """
        Convert Tool objects to Anthropic-compatible tool definitions.
        Adds cache control to tools if enabled.
        """
        tool_defs = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.json_schema["parameters"]
            }
            for tool in tools
        ]
        
        # Add cache control to the last tool definition
        if self.enable_prompt_caching and self.cache_tools and tool_defs:
            tool_defs[-1]["cache_control"] = {"type": "ephemeral"}
        
        return tool_defs

    def _process_response(self, response: Any) -> ChatLLMResponse:
        """
        Process Anthropic API response into ChatLLMResponse object.
        Includes cache usage metrics.
        """
        content_blocks = response.content
        usage_data = response.usage
        
        # Extract cache metrics
        cache_creation_tokens = getattr(usage_data, 'cache_creation_input_tokens', None) or 0
        cache_read_tokens = getattr(usage_data, 'cache_read_input_tokens', None) or 0
        
        usage = ChatLLMUsage(
            prompt_tokens=usage_data.input_tokens,
            completion_tokens=usage_data.output_tokens,
            total_tokens=usage_data.input_tokens + usage_data.output_tokens,
            cache_creation_input_tokens=cache_creation_tokens or None,
            cache_read_input_tokens=cache_read_tokens or None,
        )

        # Log cache usage
        if cache_creation_tokens > 0:
            logger.debug(f"Cache created: {cache_creation_tokens} tokens")
        if cache_read_tokens > 0:
            logger.debug(f"Cache hit: {cache_read_tokens} tokens (saved ~{cache_read_tokens * 0.9:.0f} tokens worth of cost)")
        
        text_content = ""
        thinking = None
        tool_message = None
        
        for block in content_blocks:
            if block.type == "text":
                text_content += block.text
            elif block.type == "thinking":
                thinking = block.thinking
            elif block.type == "tool_use":
                # Take the first tool use as per Agent logic
                if not tool_message:
                    tool_message = ToolMessage(
                        id=block.id,
                        name=block.name,
                        params=block.input
                    )
        
        final_content = tool_message if tool_message else AIMessage(content=text_content)
        if thinking:
            final_content.thinking = thinking
        
        return ChatLLMResponse(
            content=final_content,
            thinking=thinking,
            usage=usage
        )

    @overload
    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        ...

    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": anthropic_messages,
            "max_tokens": 4096,
            **self.kwargs
        }
        
        # Only add system if it's not None
        if system_prompt is not None:
            params["system"] = system_prompt
            
        # Only add tools if they exist
        if anthropic_tools:
            params["tools"] = anthropic_tools
            
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if structured_output:
            # Simulate structured output via tool calling
            structured_tool = {
                "name": "record_result",
                "description": f"Record the structured result: {structured_output.__name__}",
                "input_schema": structured_output.model_json_schema()
            }
            
            # Add cache control to structured output tool if caching enabled
            if self.enable_prompt_caching and self.cache_tools:
                structured_tool["cache_control"] = {"type": "ephemeral"}
            
            params["tools"] = [structured_tool]
            params["tool_choice"] = {"type": "tool", "name": "record_result"}
            
            response = self.client.messages.create(**params)
            tool_use = next(b for b in response.content if b.type == "tool_use")
            parsed = structured_output(**tool_use.input)
            
            # Extract cache metrics
            cache_creation_tokens = getattr(response.usage, 'cache_creation_input_tokens', None) or 0
            cache_read_tokens = getattr(response.usage, 'cache_read_input_tokens', None) or 0
            
            usage = ChatLLMUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                cache_creation_input_tokens=cache_creation_tokens or None,
                cache_read_input_tokens=cache_read_tokens or None,
            )

            return ChatLLMResponse(
                content=parsed,
                usage=usage,
            )

        response = self.client.messages.create(**params)
        return self._process_response(response)

    @overload
    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        ...

    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": anthropic_messages,
            "max_tokens": 4096,
            **self.kwargs
        }
        
        if system_prompt is not None:
            params["system"] = system_prompt
            
        if anthropic_tools:
            params["tools"] = anthropic_tools
            
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if structured_output:
            structured_tool = {
                "name": "record_result",
                "description": f"Record the structured result: {structured_output.__name__}",
                "input_schema": structured_output.model_json_schema()
            }
            
            if self.enable_prompt_caching and self.cache_tools:
                structured_tool["cache_control"] = {"type": "ephemeral"}
            
            params["tools"] = [structured_tool]
            params["tool_choice"] = {"type": "tool", "name": "record_result"}
            
            response = await self.aclient.messages.create(**params)
            tool_use = next(b for b in response.content if b.type == "tool_use")
            parsed = structured_output(**tool_use.input)
            
            cache_creation_tokens = getattr(response.usage, 'cache_creation_input_tokens', None) or 0
            cache_read_tokens = getattr(response.usage, 'cache_read_input_tokens', None) or 0
            
            usage = ChatLLMUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                cache_creation_input_tokens=cache_creation_tokens or None,
                cache_read_input_tokens=cache_read_tokens or None,
            )

            return ChatLLMResponse(
                content=parsed,
                usage=usage,
            )

        response = await self.aclient.messages.create(**params)
        return self._process_response(response)

    @overload
    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        ...

    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": anthropic_messages,
            "max_tokens": 4096,
            **self.kwargs
        }
        
        if system_prompt is not None:
            params["system"] = system_prompt
            
        if anthropic_tools:
            params["tools"] = anthropic_tools
            
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        with self.client.messages.stream(**params) as stream:
            for event in stream:
                if event.type == "text":
                    yield ChatLLMResponse(content=AIMessage(content=event.text))
                elif event.type == "input_json":
                    # Support partial tool call yield? 
                    # For now, we skip streaming for tool calls in Agent logic
                    pass

    @overload
    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        ...

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": anthropic_messages,
            "max_tokens": 4096,
            **self.kwargs
        }
        
        if system_prompt is not None:
            params["system"] = system_prompt
            
        if anthropic_tools:
            params["tools"] = anthropic_tools
            
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        async with self.aclient.messages.stream(**params) as stream:
            async for event in stream:
                if event.type == "text":
                    yield ChatLLMResponse(content=AIMessage(content=event.text))

    def get_metadata(self) -> Metadata:
        return Metadata(
            name=self._model,
            context_window=200000,
            owned_by="anthropic"
        )

    def get_cache_stats(self) -> dict:
        """
        Return information about the current caching configuration.
        
        Returns:
            dict: Cache configuration details
        """
        return {
            "enabled": self.enable_prompt_caching,
            "cache_system_prompt": self.cache_system_prompt,
            "cache_tools": self.cache_tools,
            "cache_recent_messages": self.cache_recent_messages,
            "model": self._model
        }