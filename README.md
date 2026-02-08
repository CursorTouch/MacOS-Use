# macOS Use Agent

An intelligent agent for macOS that uses LLMs to control your computer. Built on top of `macos-mcp`, it provides a powerful interface for automating tasks on your Mac.

## Features

- **Natural Language Control**: control your Mac using plain English commands.
- **Accessibility Tree Integration**: Uses advanced accessibility tree traversal (based on `macos-mcp`) to understand UI structure.
- **Smart Window Management**: Can launch, switch, resize, and manage application windows.
- **Robust Tooling**: Includes tools for clicking, typing, scrolling, and more, with built-in reliability features.
- **Agentic Workflow**: Uses an autonomous agent loop to plan and execute complex multi-step tasks.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Jeomon/macos-use.git
    cd macos-use
    ```

2.  Install dependencies using `uv` or `pip`:
    ```bash
    pip install -e .
    ```

3.  Configure environment variables:
    Create a `.env` file with your API keys (e.g., ANTHROPIC_API_KEY).

## Usage

Run the agent:

```bash
python main.py
```

Then enter your command, for example:
- "Open Safari and search for 'latest AI news'"
- "Organize my windows side by side"
- "Check my calendar for upcoming meetings"

## Architecture

This project integrates:
- **Agent Core**: Handles reasoning, planning, and tool execution.
- **Desktop Service**: Manages screen capture, windowing, and input simulation.
- **Tree Service**: Efficiently traverses the macOS accessibility tree to find interactive elements (imported from `macos-mcp`).

## License

MIT
