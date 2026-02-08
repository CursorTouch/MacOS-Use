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

class ChatOpenAI(BaseChatLLM):
    """
    OpenAI LLM implementation following the BaseChatLLM protocol.
    
    Supports:
    - Standard chat completions
    - Tool/function calling
    - Structured outputs (via response_format)
    - Streaming
    - Vision (image inputs)
    - Reasoning models (o1-preview, o1-mini)
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the OpenAI LLM.

        Args:
            model (str): The model name to use. Defaults to "gpt-4o".
            api_key (str, optional): OpenAI API key. Defaults to OPENAI_API_KEY environment variable.
            base_url (str, optional): Base URL for the API. Defaults to OPENAI_BASE_URL environment variable.
            timeout (float): Request timeout in seconds.
            max_retries (int): Maximum number of retries for failed requests.
            temperature (float, optional): Sampling temperature.
            **kwargs: Additional arguments to pass to the OpenAI chat completions create method.
        """
        self._model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.temperature = temperature
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.aclient = AsyncOpenAI(
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
        return "openai"
    
    def _is_reasoning_model(self) -> bool:
        """Check if the model is a reasoning model (o1 series)."""
        return self._model.startswith("o1")

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """
        Convert BaseMessage objects to OpenAI-compatible message dictionaries.
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
                # Handle AIMessage with potential thinking content
                msg_dict = {"role": "assistant", "content": msg.content}
                
                # For reasoning models, thinking might be stored differently
                if hasattr(msg, 'thinking') and msg.thinking:
                    # Store thinking in metadata or handle separately
                    # OpenAI doesn't have a standard way to pass assistant's internal reasoning
                    pass
                    
                openai_messages.append(msg_dict)
            elif isinstance(msg, ToolMessage):
                # Reconstruct the tool call and the result for history consistency
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
        Convert Tool objects to OpenAI-compatible tool definitions.
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
        Process OpenAI API response into ChatLLMResponse object.
        """
        choice = response.choices[0]
        message = choice.message
        usage_data = response.usage
        
        # Capture reasoning tokens if available (o1 models)
        reasoning_tokens = None
        if hasattr(usage_data, "completion_tokens_details") and usage_data.completion_tokens_details:
            reasoning_tokens = getattr(
                usage_data.completion_tokens_details, "reasoning_tokens", None
            )
            if reasoning_tokens is not None:
                logger.debug(f"Reasoning tokens used: {reasoning_tokens}")

        # Build usage object
        usage = ChatLLMUsage(
            prompt_tokens=usage_data.prompt_tokens,
            completion_tokens=usage_data.completion_tokens,
            total_tokens=usage_data.total_tokens,
            reasoning_tokens=reasoning_tokens,
        )
        
        # Extract thinking/reasoning content (for o1 models)
        thinking = None
        if self._is_reasoning_model():
            # For o1 models, check if there's reasoning content
            # Note: The actual field name might vary based on OpenAI API updates
            if hasattr(message, 'reasoning_content'):
                thinking = message.reasoning_content
            elif hasattr(choice, 'reasoning_content'):
                thinking = choice.reasoning_content
        
        content = None
        
        if message.tool_calls:
            # Handle tool call (picking the first one as per Agent expectations)
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
            # Handle regular completion
            content = AIMessage(content=message.content or "")
        
        # Attach thinking if available and the content object supports it
        if thinking and hasattr(content, 'thinking'):
            content.thinking = thinking
            
        return ChatLLMResponse(
            content=content,
            thinking=thinking,
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
        
        # Temperature handling - o1 models don't support temperature
        if self.temperature is not None and not self._is_reasoning_model():
            params["temperature"] = self.temperature
        
        if structured_output:
            # Use beta parse endpoint for structured outputs
            response = self.client.beta.chat.completions.parse(
                **params,
                response_format=structured_output,
            )

            reasoning_tokens = None
            if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                reasoning_tokens = getattr(
                    response.usage.completion_tokens_details, "reasoning_tokens", None
                )

            usage = ChatLLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                reasoning_tokens=reasoning_tokens,
            )

            return ChatLLMResponse(
                content=response.choices[0].message.parsed,
                usage=usage,
            )
            
        if json_mode:
            params["response_format"] = {"type": "json_object"}
            
        response = self.client.chat.completions.create(**params)
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
        
        if self.temperature is not None and not self._is_reasoning_model():
            params["temperature"] = self.temperature
        
        if structured_output:
            response = await self.aclient.beta.chat.completions.parse(
                **params,
                response_format=structured_output,
            )

            reasoning_tokens = None
            if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                reasoning_tokens = getattr(
                    response.usage.completion_tokens_details, "reasoning_tokens", None
                )

            usage = ChatLLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                reasoning_tokens=reasoning_tokens,
            )

            return ChatLLMResponse(
                content=response.choices[0].message.parsed,
                usage=usage,
            )

        if json_mode:
            params["response_format"] = {"type": "json_object"}
            
        response = await self.aclient.chat.completions.create(**params)
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
        
        if self.temperature is not None and not self._is_reasoning_model():
            params["temperature"] = self.temperature
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**params)
        
        for chunk in response:
            if not chunk.choices:
                # Final chunk with usage
                if chunk.usage:
                    reasoning_tokens = None
                    if hasattr(chunk.usage, "completion_tokens_details") and chunk.usage.completion_tokens_details:
                        reasoning_tokens = getattr(
                            chunk.usage.completion_tokens_details, "reasoning_tokens", None
                        )

                    usage = ChatLLMUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                        reasoning_tokens=reasoning_tokens,
                    )

                    yield ChatLLMResponse(usage=usage)
                continue
            
            delta = chunk.choices[0].delta
            
            if delta.content:
                yield ChatLLMResponse(content=AIMessage(content=delta.content))
            
            # Handle reasoning content for o1 models
            if self._is_reasoning_model() and hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                yield ChatLLMResponse(thinking=delta.reasoning_content)

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
        
        if self.temperature is not None and not self._is_reasoning_model():
            params["temperature"] = self.temperature
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}

        response = await self.aclient.chat.completions.create(**params)
        
        async for chunk in response:
            if not chunk.choices:
                if chunk.usage:
                    reasoning_tokens = None
                    if hasattr(chunk.usage, "completion_tokens_details") and chunk.usage.completion_tokens_details:
                        reasoning_tokens = getattr(
                            chunk.usage.completion_tokens_details, "reasoning_tokens", None
                        )

                    usage = ChatLLMUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                        reasoning_tokens=reasoning_tokens,
                    )

                    yield ChatLLMResponse(usage=usage)
                continue
            
            delta = chunk.choices[0].delta
            
            if delta.content:
                yield ChatLLMResponse(content=AIMessage(content=delta.content))
            
            if self._is_reasoning_model() and hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                yield ChatLLMResponse(thinking=delta.reasoning_content)

    def get_metadata(self) -> Metadata:
        # Determine context window based on model
        context_window = 128000  # Default for GPT-4o and newer models
        
        if self._model.startswith("gpt-4-turbo"):
            context_window = 128000
        elif self._model.startswith("gpt-4"):
            context_window = 8192
        elif self._model.startswith("gpt-3.5-turbo"):
            context_window = 16385
        elif self._model.startswith("o1"):
            context_window = 200000  # o1 models have larger context
            
        return Metadata(
            name=self._model,
            context_window=context_window,
            owned_by="openai"
        )