import os
import json
import logging
from typing import Iterator, AsyncIterator, List, Optional, Any, Union, overload
from openai import AzureOpenAI, AsyncAzureOpenAI
from pydantic import BaseModel
from macos_use.llms.base import BaseChatLLM
from macos_use.llms.views import ChatLLMResponse, ChatLLMUsage, Metadata
from macos_use.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ImageMessage, ToolMessage
from macos_use.tool import Tool

logger = logging.getLogger(__name__)

class ChatAzureOpenAI(BaseChatLLM):
    """
    Azure OpenAI LLM implementation following the BaseChatLLM protocol.
    """
    
    def __init__(
        self,
        deployment_name: str,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-02-01",
        timeout: float = 600.0,
        max_retries: int = 2,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the Azure OpenAI LLM.

        Args:
            deployment_name (str): The Azure deployment name to use.
            api_key (str, optional): Azure API key. Defaults to AZURE_OPENAI_API_KEY.
            azure_endpoint (str, optional): Azure endpoint URL. Defaults to AZURE_OPENAI_ENDPOINT.
            api_version (str): Azure API version.
            timeout (float): Request timeout.
            max_retries (int): Maximum retries.
            temperature (float, optional): Sampling temperature.
            **kwargs: Additional arguments for chat completions.
        """
        self._deployment = deployment_name
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.temperature = temperature
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=api_version,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.aclient = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=api_version,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._deployment

    @property
    def provider(self) -> str:
        return "azure_openai"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """
        Convert BaseMessage objects to Azure-compatible message dictionaries.
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
        Convert Tool objects to Azure-compatible tool definitions.
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
        Process Azure API response into ChatLLMResponse object.
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
        openai_messages = self._convert_messages(messages)
        openai_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._deployment,
            "messages": openai_messages,
            **self.kwargs
        }
        
        # Only add tools if they exist
        if openai_tools:
            params["tools"] = openai_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if structured_output:
            # Use beta parse endpoint for structured outputs
            response = self.client.beta.chat.completions.parse(
                **params,
                response_format=structured_output
            )
            
            usage = ChatLLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
            
            return ChatLLMResponse(
                content=response.choices[0].message.parsed,
                usage=usage
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
            "model": self._deployment,
            "messages": openai_messages,
            **self.kwargs
        }
        
        if openai_tools:
            params["tools"] = openai_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if structured_output:
            response = await self.aclient.beta.chat.completions.parse(
                **params,
                response_format=structured_output
            )
            
            usage = ChatLLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
            
            return ChatLLMResponse(
                content=response.choices[0].message.parsed,
                usage=usage
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
            "model": self._deployment,
            "messages": openai_messages,
            "stream": True,
            **self.kwargs
        }
        
        if openai_tools:
            params["tools"] = openai_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**params)
        
        for chunk in response:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            
            if delta.content:
                yield ChatLLMResponse(content=AIMessage(content=delta.content))

    @overload
    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        ...

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        openai_messages = self._convert_messages(messages)
        openai_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._deployment,
            "messages": openai_messages,
            "stream": True,
            **self.kwargs
        }
        
        if openai_tools:
            params["tools"] = openai_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        
        response = await self.aclient.chat.completions.create(**params)
        
        async for chunk in response:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            
            if delta.content:
                yield ChatLLMResponse(content=AIMessage(content=delta.content))

    def get_metadata(self) -> Metadata:
        return Metadata(
            name=self._deployment,
            context_window=128000,
            owned_by="azure_openai"
        )
