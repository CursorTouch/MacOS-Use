from macos_use.messages import AIMessage, ToolMessage
from pydantic import BaseModel


class ChatLLMUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    image_tokens: int | None = None
    thinking_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class ChatLLMResponse(BaseModel):
    content: AIMessage | ToolMessage | BaseModel | None = None
    thinking: str | None = None
    thinking_signature: str | bytes | None = None
    usage: ChatLLMUsage | None = None


class Metadata(BaseModel):
    name: str
    context_window: int
    owned_by: str