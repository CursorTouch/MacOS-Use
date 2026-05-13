<div align="center">
  <h1>🍎 macOS-Use</h1>

  <a href="https://github.com/CursorTouch/MacOS-Use/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
  <img src="https://img.shields.io/pypi/dm/macos-use" alt="PyPI Downloads">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/platform-macOS%2012%2B-blue" alt="Platform: macOS 12+">
  <img src="https://img.shields.io/github/last-commit/CursorTouch/MacOS-Use" alt="Last Commit">
  <a href="https://x.com/CursorTouch">
    <img src="https://img.shields.io/badge/follow-%40CursorTouch-1DA1F2?logo=twitter&style=flat" alt="Follow on Twitter">
  </a>
  <a href="https://discord.com/invite/Aue9Yj2VzS">
    <img src="https://img.shields.io/badge/Join%20on-Discord-5865F2?logo=discord&logoColor=white&style=flat" alt="Join us on Discord">
  </a>
</div>

MacOS-Use is an AI agent that controls macOS at the GUI layer. It reads the screen via the macOS Accessibility API and uses any LLM to decide what to click, type, scroll, or run — no computer vision model required.

Give it a task in plain English. It handles the rest.

## What It Can Do

- Open, switch between, and resize application windows
- Click, type, scroll, drag, and use keyboard shortcuts
- Run shell commands and AppleScript via `osascript`
- Scrape web pages via the browser accessibility tree
- Read and write files on the filesystem
- Manage macOS virtual desktops (Spaces) via Mission Control
- Remember information across steps with persistent memory
- Speak and listen via STT/TTS (voice input and output)

## Installation

Prerequisites: Python 3.12+, macOS 12 (Monterey) or later

```bash
pip install macos-use
```

Or with `uv`:

```bash
uv add macos-use
```

## Quick Start

Pick any supported LLM provider and run a task:

### Anthropic (Claude)

```python
from macos_use.providers.anthropic import ChatAnthropic
from macos_use import Agent, Browser

llm = ChatAnthropic(model="claude-sonnet-4-5")
agent = Agent(llm=llm, browser=Browser.SAFARI)
agent.invoke(task="Open Notes and write a short poem about macOS")
```

### OpenAI

```python
from macos_use.providers.openai import ChatOpenAI
from macos_use import Agent, Browser

llm = ChatOpenAI(model="gpt-4o")
agent = Agent(llm=llm, browser=Browser.CHROME)
agent.invoke(task="Search for the weather in New York on Google")
```

### Google Gemini

```python
from macos_use.providers.google import ChatGoogle
from macos_use import Agent, Browser

llm = ChatGoogle(model="gemini-2.5-flash")
agent = Agent(llm=llm, browser=Browser.SAFARI)
agent.invoke(task=input("Enter a task: "))
```

### Ollama (Local)

```python
from macos_use.providers.ollama import ChatOllama
from macos_use import Agent

llm = ChatOllama(model="qwen3-vl:4b")
agent = Agent(llm=llm, use_vision=False)
agent.invoke(task=input("Enter a task: "))
```

### Async Usage

```python
import asyncio
from macos_use.providers.anthropic import ChatAnthropic
from macos_use import Agent

async def main():
    llm = ChatAnthropic(model="claude-sonnet-4-5")
    agent = Agent(llm=llm)
    result = await agent.ainvoke(task="Take a screenshot and describe the desktop")
    print(result.content)

asyncio.run(main())
```

## CLI

Run the interactive agent directly from your terminal:

```bash
macos-use
```

## Supported LLM Providers

| Provider | Import |
| --- | --- |
| Anthropic | `from macos_use.providers.anthropic import ChatAnthropic` |
| OpenAI | `from macos_use.providers.openai import ChatOpenAI` |
| Google | `from macos_use.providers.google import ChatGoogle` |
| Groq | `from macos_use.providers.groq import ChatGroq` |
| Ollama | `from macos_use.providers.ollama import ChatOllama` |
| Mistral | `from macos_use.providers.mistral import ChatMistral` |
| Cerebras | `from macos_use.providers.cerebras import ChatCerebras` |
| DeepSeek | `from macos_use.providers.deepseek import ChatDeepSeek` |
| Azure OpenAI | `from macos_use.providers.azure_openai import ChatAzureOpenAI` |
| Open Router | `from macos_use.providers.open_router import ChatOpenRouter` |
| LiteLLM | `from macos_use.providers.litellm import ChatLiteLLM` |
| NVIDIA | `from macos_use.providers.nvidia import ChatNvidia` |
| vLLM | `from macos_use.providers.vllm import ChatVLLM` |

## Agent Configuration

```python
Agent(
    llm=llm,                        # LLM instance (required)
    mode="normal",                  # "normal" (full context) or "flash" (lightweight, faster)
    browser=Browser.SAFARI,         # Browser.SAFARI | Browser.CHROME | Browser.FIREFOX | Browser.EDGE
    use_vision=False,               # Send screenshots to the LLM
    use_annotation=False,           # Annotate UI elements on screenshots
    use_accessibility=True,         # Use the macOS accessibility tree
    auto_minimize=False,            # Minimize active window before the agent starts
    max_steps=25,                   # Max number of steps before giving up
    max_consecutive_failures=3,     # Abort after N consecutive tool failures
    instructions=[],                # Extra system instructions
    log_to_console=True,            # Print steps to the console
    log_to_file=False,              # Write steps to a log file
    event_subscriber=None,          # Custom event listener (see Events section)
    experimental=False,             # Enable experimental tools (memory, multi-select, multi-edit)
)
```

Tip: Use `claude-haiku-4-5`, `claude-sonnet-4-5`, or `claude-opus-4-5` for best results.

## Tools

The agent has access to these tools automatically — no configuration needed.

### Core Tools

| Tool | Description |
| --- | --- |
| `click_tool` | Left, right, middle click or hover at coordinates |
| `type_tool` | Type text into any input field |
| `scroll_tool` | Scroll vertically or horizontally |
| `move_tool` | Move mouse or drag-and-drop |
| `shortcut_tool` | Press keyboard shortcuts (e.g. `cmd+c`, `cmd+tab`) |
| `app_tool` | Launch, switch, or resize application windows |
| `shell_tool` | Run bash commands or AppleScript (`osascript`) |
| `scrape_tool` | Extract text content from web pages |
| `desktop_tool` | Create, remove, switch macOS virtual desktops (Spaces) |
| `wait_tool` | Pause execution for N seconds |
| `done_tool` | Return the final answer to the user |

### Experimental Tools

Enable with `experimental=True`.

| Tool | Description |
| --- | --- |
| `memory_tool` | Persist information across steps in markdown files |
| `multi_select_tool` | Cmd+click multiple elements at once |
| `multi_edit_tool` | Fill multiple form fields in one action |

## Events

Observe every step the agent takes with the event system:

```python
from macos_use import Agent, AgentEvent, EventType, BaseEventSubscriber

class MySubscriber(BaseEventSubscriber):
    def invoke(self, event: AgentEvent):
        if event.type == EventType.TOOL_CALL:
            print(f"Tool: {event.data['tool_name']}")
        elif event.type == EventType.DONE:
            print(f"Done: {event.data['content']}")

agent = Agent(llm=llm, event_subscriber=MySubscriber())
```

Or use a plain callable:

```python
def on_event(event: AgentEvent):
    print(f"{event.type.value}: {event.data}")

agent = Agent(llm=llm, event_subscriber=on_event)
```

Event types: `THOUGHT` · `TOOL_CALL` · `TOOL_RESULT` · `DONE` · `ERROR`

## Voice (STT / TTS)

MacOS-Use supports voice input and spoken output via multiple providers.

STT (Speech-to-Text): OpenAI Whisper · Google · Groq · ElevenLabs · Deepgram

TTS (Text-to-Speech): OpenAI · Google · Groq · ElevenLabs · Deepgram

```python
from macos_use.providers.openai import ChatOpenAI, STTOpenAI, TTSOpenAI
from macos_use.speech import STT, TTS

llm = ChatOpenAI(model="gpt-4o")
stt = STT(provider=STTOpenAI())
tts = TTS(provider=TTSOpenAI())

task = stt.invoke()
agent = Agent(llm=llm)
result = agent.invoke(task=task)
tts.invoke(result.content)
```

## Virtual Desktops (Spaces)

The agent can manage macOS Spaces natively via Mission Control:

```python
agent.invoke(task="Create a new Space and switch to it")
agent.invoke(task="Switch to Space 2")
agent.invoke(task="Remove the current Space")
```

Or use `desktop_tool` directly with actions: `create`, `remove`, `switch`.

Note: Switching by number requires the keyboard shortcut to be enabled in:

`System Settings → Keyboard → Shortcuts → Mission Control`

## Security

This agent can:

- Operate your computer on behalf of the user
- Modify files and system settings
- Make irreversible changes to your system

STRONGLY RECOMMENDED: Deploy in a Virtual Machine or dedicated test machine

The project provides no sandbox or isolation layer. For your safety:

- Use a Virtual Machine (`UTM`, `Parallels`, `VMware Fusion`)
- Use a dedicated test Mac
- Close sensitive applications before running

Read the full Security Policy before deployment.

## Telemetry

MacOS-Use includes lightweight, privacy-friendly telemetry to help improve reliability and understand real-world usage.

Disable it at any time:

```bash
ANONYMIZED_TELEMETRY=false
```

Or in code:

```python
import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"
```

## License

MIT — see `LICENSE`.

## Acknowledgements

- `PyObjC` — macOS Accessibility API bindings

## Contributing

Contributions are welcome. See `CONTRIBUTING` for the development workflow.

Made with love by Jeomon George

## Citation

```bibtex
@software{
  author       = {George, Jeomon},
  title        = {MacOS-Use: Enable AI to control macOS},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/CursorTouch/MacOS-Use}
}
```
