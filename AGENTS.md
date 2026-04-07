# MacOS Use

## Description

MacOS Use is an AI Agent that interacts with macOS at the GUI level — computer use for the macOS operating system. The project operates on top of the macOS Accessibility API and the `macos-mcp` integration layer.

## How it works

MacOS Use uses a combination of tools to interact with the macOS. It uses the accessibility tree to get the UI elements and then uses the tools to interact with those elements.

## Tools

MacOS Use has the following tools:

- **App Tool**: Manages macOS applications (launch, resize, switch).
- **Click Tool**: Performs mouse clicks (left, right, middle, single, double) at specific coordinates.
- **Drag Tool**: Performs drag-and-drop operations from the current cursor location to a destination.
- **Done Tool**: Signals the successful completion of a task and returns the final answer.
- **Memory Tool**: Provides persistent file-based storage to read, write, update, delete, and view memory files. Useful for saving context and sharing data.
- **File Tool**: Performs file system operations: read, write, list, delete, move, copy, or check existence.
- **Move Tool**: Moves the mouse cursor to specific coordinates without clicking (hovering).
- **Scrape Tool**: Fetches webpage content and converts it to markdown via the browser accessibility tree.
- **Scroll Tool**: Scrolls content vertically or horizontally.
- **Shell Tool**: Executes shell commands and returns output with status codes.
- **Shortcut Tool**: Executes keyboard shortcuts (e.g., `cmd+c`, `cmd+tab`).
- **Type Tool**: Types text into focused UI elements, with options to clear text or press Enter.
- **Wait Tool**: Pauses execution for a specified duration to allow processes or UI animations to complete.

## Agent Config

- `mode`: the agent has two modes, `flash` and `normal`.
  - `flash`: The agent will use the flash prompt to generate the response. (Lightweight)
  - `normal`: The agent will use the normal prompt to generate the response.
- `max_steps`: Maximum number of steps the agent can take to complete the task.
- `llm`: LLM to be used by the agent.
- `use_vision`: Screenshot will be given to the LLM if true.
- `use_annotation`: Annotated screenshot will be given to the LLM if true.
- `auto_minimize`: Minimize the window of the agent running interface (temporarily).

## Key Commands

The project uses `uv` as the package manager.
- `uv sync` to install dependencies and create virtual environment.
- `uv run main.py` to start the agent.
- `uv add macos-use` to add the package to an existing project.

## Testing

- `pytest` to run all tests.

## Project Structure

- `macos_use/llms` — To access different LLMs.
- `macos_use/agent` — Core Agent implementation (includes tools, prompts, etc.)
- `macos_use/agent/tools` — Tools used by the agent to interact with macOS.
- `macos_use/agent/prompt` — Managing the prompts of the agent (system_prompt, observation, etc.)
- `macos_use/agent/desktop` — Desktop Module containing high-level information about macOS (Windows, Desktop, Dock, Menu Bar, etc.)
- `macos_use/agent/tree` — Tree Module containing information about UI elements in Windows or Apps.
- `macos_use/agent/registry` — Registry contains the tools owned by the agent.
- `macos_use/tool` — The Tool Module used to implement the tools for the agent.
- `macos_use/messages` — Messages Module used to implement various message formats for the LLM (SystemMessage, HumanMessage, AIMessage, etc.)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Security

- This project has the ability to operate macOS at GUI level. Be careful while using this project.
- Best practice: launch the project in a virtual machine so that if something goes wrong, you can revert the changes.
