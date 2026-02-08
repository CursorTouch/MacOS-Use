import os
import uuid
import logging
from typing import Iterator, AsyncIterator, List, Optional, Any, overload
from google import genai
from google.genai import types
from pydantic import BaseModel
from macos_use.llms.base import BaseChatLLM
from macos_use.llms.views import ChatLLMResponse, ChatLLMUsage, Metadata
from macos_use.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ImageMessage, ToolMessage
from macos_use.tool import Tool

logger = logging.getLogger(__name__)


class ChatGoogle(BaseChatLLM):
    """
    Google Gemini LLM implementation following the BaseChatLLM protocol.

    Supports:
    - Standard chat completions (generate_content)
    - Tool/function calling
    - Structured outputs (via response_schema)
    - Streaming (sync and async)
    - Vision (image inputs)
    - Thinking models (Gemini 2.5 with extended thinking)
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking_budget: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the Google Gemini LLM.

        Args:
            model: The model name to use. Defaults to "gemini-2.5-flash".
            api_key: Google API key. Falls back to GEMINI_API_KEY or GOOGLE_API_KEY env vars.
            temperature: Sampling temperature.
            thinking_budget: Token budget for extended thinking (Gemini 2.5 models).
            **kwargs: Additional arguments for GenerateContentConfig.
        """
        self._model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.temperature = temperature
        self.thinking_budget = thinking_budget

        self.client = genai.Client(api_key=self.api_key)
        self.kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "google"

    def _is_thinking_model(self) -> bool:
        """Check if the model supports extended thinking (Gemini 2.5 series)."""
        return "2.5" in self._model

    def _convert_messages(
        self, messages: List[BaseMessage]
    ) -> tuple[Optional[str], list[types.Content]]:
        """
        Convert BaseMessage objects to Google Gemini-compatible content list.

        Gemini requires strict user/model role alternation. This method builds
        the raw content list and then merges consecutive same-role entries so
        that, e.g., a function_response (user) followed by an observation
        HumanMessage (user) become a single user Content with combined parts.

        Returns:
            A tuple of (system_instruction, contents).
        """
        raw_contents: list[types.Content] = []
        system_instruction: Optional[str] = None

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_instruction = msg.content
            elif isinstance(msg, HumanMessage):
                raw_contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )
            elif isinstance(msg, ImageMessage):
                parts: list[types.Part] = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))

                img_bytes_list = msg.convert_images(format="bytes")
                for img_bytes in img_bytes_list:
                    parts.append(
                        types.Part.from_bytes(data=img_bytes, mime_type=msg.mime_type)
                    )
                raw_contents.append(types.Content(role="user", parts=parts))
            elif isinstance(msg, AIMessage):
                parts = []
                if msg.thinking:
                    parts.append(types.Part(thought=True, text=msg.thinking))
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                if parts:
                    raw_contents.append(types.Content(role="model", parts=parts))
            elif isinstance(msg, ToolMessage):
                # Model's function call (with optional thinking)
                model_parts: list[types.Part] = []
                if msg.thinking:
                    model_parts.append(types.Part(thought=True, text=msg.thinking))
                model_parts.append(
                    types.Part.from_function_call(name=msg.name, args=msg.params)
                )
                raw_contents.append(types.Content(role="model", parts=model_parts))

                # Function response
                raw_contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=msg.name,
                                response={"result": msg.content or ""},
                            )
                        ],
                    )
                )

        # Merge consecutive same-role contents to satisfy Gemini's strict
        # user/model alternation requirement.
        contents = self._merge_consecutive_contents(raw_contents)
        return system_instruction, contents

    @staticmethod
    def _merge_consecutive_contents(
        contents: list[types.Content],
    ) -> list[types.Content]:
        """
        Merge consecutive Content objects that share the same role.

        Gemini requires strict role alternation (user → model → user → …).
        After converting ToolMessage into a model turn (function_call) and a
        user turn (function_response), the next HumanMessage also becomes a
        user turn, creating back-to-back user entries. This helper collapses
        them into a single Content with combined parts.
        """
        if not contents:
            return contents

        merged: list[types.Content] = [contents[0]]
        for content in contents[1:]:
            if content.role == merged[-1].role:
                # Combine parts into the existing Content
                merged[-1] = types.Content(
                    role=content.role,
                    parts=(merged[-1].parts or []) + (content.parts or []),
                )
            else:
                merged.append(content)
        return merged

    def _convert_tools(self, tools: List[Tool]) -> list[types.Tool]:
        """
        Convert Tool objects to Google Gemini-compatible tool definitions.
        """
        function_declarations = []
        for tool in tools:
            schema = tool.json_schema
            function_declarations.append(
                types.FunctionDeclaration(
                    name=schema["name"],
                    description=schema.get("description", ""),
                    parameters_json_schema=schema.get("parameters", {}),
                )
            )
        return [types.Tool(function_declarations=function_declarations)]

    def _build_config(
        self,
        system_instruction: Optional[str],
        tools: Optional[list[types.Tool]],
        structured_output: Optional[type[BaseModel]] = None,
        json_mode: bool = False,
    ) -> types.GenerateContentConfig:
        """
        Build the GenerateContentConfig for the API call.
        """
        config_params: dict[str, Any] = {}

        if system_instruction:
            config_params["system_instruction"] = system_instruction

        if tools:
            config_params["tools"] = tools
            config_params["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )

        if self.temperature is not None:
            config_params["temperature"] = self.temperature

        # Thinking budget for 2.5 models
        if self._is_thinking_model() and self.thinking_budget is not None:
            config_params["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )

        if structured_output:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = structured_output
        elif json_mode:
            config_params["response_mime_type"] = "application/json"

        return types.GenerateContentConfig(**config_params)

    def _extract_usage(self, usage_metadata: Any) -> ChatLLMUsage:
        """
        Extract usage information from the response metadata.
        """
        if not usage_metadata:
            return ChatLLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        thinking_tokens = getattr(usage_metadata, "thoughts_token_count", None)

        return ChatLLMUsage(
            prompt_tokens=usage_metadata.prompt_token_count or 0,
            completion_tokens=usage_metadata.candidates_token_count or 0,
            total_tokens=usage_metadata.total_token_count or 0,
            thinking_tokens=thinking_tokens,
        )

    def _extract_thinking(self, response: Any) -> Optional[str]:
        """
        Extract thinking/reasoning content from the response parts.
        """
        thinking_parts: list[str] = []
        try:
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if getattr(part, "thought", False) and part.text:
                        thinking_parts.append(part.text)
        except (AttributeError, IndexError):
            pass
        return "\n".join(thinking_parts) if thinking_parts else None

    def _process_response(self, response: Any) -> ChatLLMResponse:
        """
        Process Google Gemini API response into ChatLLMResponse object.
        """
        usage = self._extract_usage(response.usage_metadata)

        # Check for function calls first
        function_calls = response.function_calls
        if function_calls:
            fc = function_calls[0]
            fc_id = getattr(fc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"

            content = ToolMessage(
                id=fc_id,
                name=fc.name,
                params=dict(fc.args) if fc.args else {},
            )

            thinking = self._extract_thinking(response)
            if thinking:
                content.thinking = thinking

            return ChatLLMResponse(
                content=content,
                thinking=thinking,
                usage=usage,
            )

        # Handle regular text response
        try:
            text_content = response.text or ""
        except (ValueError, AttributeError):
            text_content = ""

        thinking = self._extract_thinking(response)

        content = AIMessage(content=text_content)
        if thinking:
            content.thinking = thinking

        return ChatLLMResponse(
            content=content,
            thinking=thinking,
            usage=usage,
        )

    @overload
    def invoke(
        self,
        messages: list[BaseMessage],
        tools: list[Tool] = [],
        structured_output: BaseModel | None = None,
        json_mode: bool = False,
    ) -> ChatLLMResponse: ...

    def invoke(
        self,
        messages: list[BaseMessage],
        tools: list[Tool] = [],
        structured_output: BaseModel | None = None,
        json_mode: bool = False,
    ) -> ChatLLMResponse:
        system_instruction, contents = self._convert_messages(messages)
        google_tools = self._convert_tools(tools) if tools else None
        config = self._build_config(system_instruction, google_tools, structured_output, json_mode)

        response = self.client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        if structured_output:
            parsed = structured_output.model_validate_json(response.text)
            usage = self._extract_usage(response.usage_metadata)
            return ChatLLMResponse(content=parsed, usage=usage)

        return self._process_response(response)

    @overload
    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools: list[Tool] = [],
        structured_output: BaseModel | None = None,
        json_mode: bool = False,
    ) -> ChatLLMResponse: ...

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools: list[Tool] = [],
        structured_output: BaseModel | None = None,
        json_mode: bool = False,
    ) -> ChatLLMResponse:
        system_instruction, contents = self._convert_messages(messages)
        google_tools = self._convert_tools(tools) if tools else None
        config = self._build_config(system_instruction, google_tools, structured_output, json_mode)

        response = await self.client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        if structured_output:
            parsed = structured_output.model_validate_json(response.text)
            usage = self._extract_usage(response.usage_metadata)
            return ChatLLMResponse(content=parsed, usage=usage)

        return self._process_response(response)

    @overload
    def stream(
        self,
        messages: list[BaseMessage],
        tools: list[Tool] = [],
        structured_output: BaseModel | None = None,
        json_mode: bool = False,
    ) -> Iterator[ChatLLMResponse]: ...

    def stream(
        self,
        messages: list[BaseMessage],
        tools: list[Tool] = [],
        structured_output: BaseModel | None = None,
        json_mode: bool = False,
    ) -> Iterator[ChatLLMResponse]:
        system_instruction, contents = self._convert_messages(messages)
        google_tools = self._convert_tools(tools) if tools else None
        config = self._build_config(system_instruction, google_tools, structured_output, json_mode)

        for chunk in self.client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        ):
            # Yield thinking parts
            try:
                if chunk.candidates and chunk.candidates[0].content:
                    for part in chunk.candidates[0].content.parts:
                        if getattr(part, "thought", False) and part.text:
                            yield ChatLLMResponse(thinking=part.text)
            except (AttributeError, IndexError):
                pass

            # Yield text content (chunk.text excludes thought parts)
            try:
                if chunk.text:
                    yield ChatLLMResponse(content=AIMessage(content=chunk.text))
            except (AttributeError, ValueError):
                pass

            # Yield usage on final chunk
            if chunk.usage_metadata and chunk.usage_metadata.total_token_count:
                yield ChatLLMResponse(usage=self._extract_usage(chunk.usage_metadata))

    @overload
    async def astream(
        self,
        messages: list[BaseMessage],
        tools: list[Tool] = [],
        structured_output: BaseModel | None = None,
        json_mode: bool = False,
    ) -> AsyncIterator[ChatLLMResponse]: ...

    async def astream(
        self,
        messages: list[BaseMessage],
        tools: list[Tool] = [],
        structured_output: BaseModel | None = None,
        json_mode: bool = False,
    ) -> AsyncIterator[ChatLLMResponse]:
        system_instruction, contents = self._convert_messages(messages)
        google_tools = self._convert_tools(tools) if tools else None
        config = self._build_config(system_instruction, google_tools, structured_output, json_mode)

        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        ):
            # Yield thinking parts
            try:
                if chunk.candidates and chunk.candidates[0].content:
                    for part in chunk.candidates[0].content.parts:
                        if getattr(part, "thought", False) and part.text:
                            yield ChatLLMResponse(thinking=part.text)
            except (AttributeError, IndexError):
                pass

            # Yield text content (chunk.text excludes thought parts)
            try:
                if chunk.text:
                    yield ChatLLMResponse(content=AIMessage(content=chunk.text))
            except (AttributeError, ValueError):
                pass

            # Yield usage on final chunk
            if chunk.usage_metadata and chunk.usage_metadata.total_token_count:
                yield ChatLLMResponse(usage=self._extract_usage(chunk.usage_metadata))

    def get_metadata(self) -> Metadata:
        context_window = 1048576  # Default for Gemini 2.x models (1M tokens)

        if "1.0" in self._model:
            context_window = 32768
        elif "1.5" in self._model:
            context_window = 1048576
        elif "2.0" in self._model or "2.5" in self._model:
            context_window = 1048576

        return Metadata(
            name=self._model,
            context_window=context_window,
            owned_by="google",
        )
