from macos_use.messages import SystemMessage, HumanMessage, AIMessage, ImageMessage, ToolMessage
from macos_use.agent.tools import BUILTIN_TOOLS, EXPERIMENTAL_TOOLS
from macos_use.agent.views import AgentResult, AgentState
from macos_use.telemetry.service import ProductTelemetry
from macos_use.agent.registry.service import Registry
from macos_use.agent.watchdog.service import WatchDog
from macos_use.agent.registry.views import ToolResult
from macos_use.agent.desktop.service import Desktop
from macos_use.agent.desktop.views import Browser
from macos_use.agent.prompt.service import Prompt
from macos_use.llms.base import BaseChatLLM
from macos_use.agent.base import BaseAgent
from contextlib import nullcontext
from rich.console import Console
from datetime import datetime
from typing import Literal
from pathlib import Path
import logging
import time

logger = logging.getLogger("macos_use")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def _create_file_logger() -> logging.FileHandler:
    """Create a per-run file handler that logs to logs/<timestamp>.log"""
    logs_dir = Path(__file__).resolve().parents[2] / "logs"
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(file_formatter)
    return file_handler


class Agent(BaseAgent):
    def __init__(
        self,
        mode: Literal["flash", "normal"] = "normal",
        instructions: list[str] | None = None,
        browser: Browser = Browser.EDGE,
        use_annotation: bool = False,
        use_accessibility: bool = True,
        secrets: dict[str, str] = {},
        llm: BaseChatLLM = None,
        max_consecutive_failures: int = 3,
        max_steps: int = 25,
        use_vision: bool = False,
        auto_minimize: bool = False,
        log_to_file:bool = False,
        experimental: bool = False,
    ):
        """Initialize the Agent.

        The Agent is the core component that orchestrates interactions with the macOS GUI.
        It uses an LLM to process instructions, analyze the desktop state (via UI automation
        and optionally vision), and execute tools to achieve the desired goals.

        Args:
            mode: Agent mode - "flash" for lightweight prompts, "normal" for full prompts.
            instructions: A list of additional instructions or goals for the agent to execute.
            browser: The target web browser for web-related tasks. Defaults to Browser.EDGE.
            use_annotation: Whether to overlay UI element annotations on screenshots before
                providing to the LLM. Defaults to False.
            use_accessibility: Whether to use the accessibility tree. Defaults to True.
            secrets: A dictionary of secret key-value pairs accessible to the agent.
            llm: The Large Language Model instance used for decision making.
            max_consecutive_failures: Maximum number of consecutive failures before giving up.
            max_steps: Maximum number of steps allowed in the agent's execution.
            use_vision: Whether to provide screenshots to the LLM. Defaults to False.
            auto_minimize: Whether to automatically minimize the current window before agent
                proceeds. Defaults to False.
            experimental: Whether to include experimental tools. Defaults to False.
        """
        self.name = "MacOS-Use"
        self.description = "An agent that can interact with GUI elements on macOS"
        self.mode = mode
        self.secrets = secrets
        self.registry = Registry(
            BUILTIN_TOOLS + EXPERIMENTAL_TOOLS if experimental else BUILTIN_TOOLS
        )
        self.instructions = instructions or []
        self.browser = browser
        self.auto_minimize = auto_minimize
        if use_annotation and not use_vision:
            logger.warning("use_vision is set to True if use_annotation is True.")
        if use_annotation and not use_accessibility:
            logger.warning("use_accessibility is set to True if use_annotation is True.")
        self.desktop = Desktop(
            use_vision=True if use_annotation else use_vision,
            use_annotation=use_annotation,
            use_accessibility=True if use_annotation else use_accessibility,
        )
        self.state = AgentState(
            max_consecutive_failures=max_consecutive_failures,
            max_steps=max_steps,
        )
        self.telemetry = ProductTelemetry()
        self.watchdog = WatchDog()
        self.console = Console()
        self.prompt = Prompt()
        self.log_to_file = log_to_file 
        self.llm = llm

    @property
    def system_message(self) -> SystemMessage:
        content = self.prompt.system(
            mode=self.mode,
            desktop=self.desktop,
            browser=self.browser,
            max_steps=self.state.max_steps,
            instructions=self.instructions,
        )
        return SystemMessage(content=content)

    @property
    def tools(self):
        return self.registry.get_tools()

    @property
    def state_message(self) -> HumanMessage | ImageMessage:
        desktop_state = self.desktop.get_state()
        content = self.prompt.human(
            query=self.state.task,
            step=self.state.step,
            max_steps=self.state.max_steps,
            desktop=self.desktop,
        )
        if self.desktop.use_vision and desktop_state.screenshot:
            image = desktop_state.screenshot
            return ImageMessage(image=image, content=content)
        return HumanMessage(content=content)

    def reason(self) -> ToolMessage:
        """Call the LLM and return the response message.

        Retries up to max_consecutive_failures times with exponential backoff on failure.

        Returns:
            The LLM response as a ToolMessage.

        Raises:
            ValueError: If the LLM returns an unexpected response type.
            Exception: If all retry attempts are exhausted.
        """
        max_attempts = self.state.max_consecutive_failures
        last_error = None

        for attempt in range(max_attempts):
            try:
                llm_response = self.llm.invoke(messages=self.state.messages+self.state.error_messages, tools=self.tools)
                content = llm_response.content

                if content is None:
                    raise ValueError(
                        f"LLM returned None content (provider: {self.llm.provider}, "
                        f"model: {self.llm.model_name})"
                    )
                elif isinstance(content, AIMessage):
                    feedback_message=HumanMessage(content="Response rejected, please use the `done_tool` to respond to the user.")
                    self.state.error_messages.extend([content,feedback_message])
                    continue
                elif isinstance(content, ToolMessage):
                    self.state.error_messages.clear()
                    return content
                else:       
                    raise ValueError(
                        f"LLM returned unexpected content type: {type(content).__name__}. "
                        f"Expected ToolMessage or AIMessage."
                    )

            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.error(
                        f"Failed to get response from {self.llm.provider} "
                        f"for {self.llm.model_name}.\n"
                        f"Retrying...({attempt + 1}/{max_attempts})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to get response from {self.llm.provider} "
                        f"for {self.llm.model_name}.\n"
                        f"All {max_attempts} attempts exhausted."
                    )

        if last_error is not None:
            raise last_error
        raise RuntimeError(
            f"LLM failed to return a tool call after {max_attempts} attempts "
            f"(provider: {self.llm.provider}, model: {self.llm.model_name})"
        )

    def act(self, tool_name: str, tool_params: dict) -> ToolResult:
        tool_result = self.registry.execute(
            tool_name=tool_name, tool_params=tool_params, desktop=self.desktop
        )
        return tool_result

    def result(self, content: str, is_done: bool = True) -> AgentResult:
        return AgentResult(content=content, is_done=is_done)

    def loop(self) -> AgentResult:
        """Run the main agent loop.

        Iterates up to max_steps, calling the LLM and executing tools each step.
        Tracks consecutive tool failures and aborts if the threshold is reached.

        Returns:
            AgentResult with the final answer or an error/timeout indication.
        """
        self.state.messages.insert(0, self.system_message)
        consecutive_failures = 0

        for step in range(self.state.max_steps):
            self.state.step = step
            logger.info(f"[Agent] 👁️ Observing: Scanning desktop...")
            self.state.messages.append(self.state_message)

            logger.info(f"[Agent] 🧠 Thinking: Reasoning...")
            try:
                message = self.reason()
            except Exception as e:
                logger.error(f"[Agent] Step {step + 1}/{self.state.max_steps} - Reason failed: {e}")
                return AgentResult(
                    is_done=False,
                    error=f"Agent failed after exhausting retries: {e}",
                )

            self.state.messages.pop()  # Remove the state message from the messages list

            tool_name = message.name
            tool_params = message.params

            evaluate = tool_params.get('evaluate', 'neutral')
            logger.info(f"[Agent] 📊 Evaluate: {evaluate}")
            logger.info(f"[Agent] 🧠 Thinking: {tool_params.get('thought', '')}")

            if tool_name != "done_tool":
                formatted_params = [
                    f"{key}={value}"
                    for key, value in tool_params.items()
                    if key not in ["thought", "evaluate"]
                ]
                logger.info(
                    f"[Agent] 🛠️ Tool Call: {tool_name}({', '.join(formatted_params)})"
                )

            tool_result = self.act(tool_name=tool_name, tool_params=tool_params)
            if tool_result.is_success:
                content = tool_result.content
                message.content = content
                self.state.messages.append(message)
            else:
                content = tool_result.error
                message.content = content
                self.state.error_messages.append(message)

            # Track consecutive failures
            if not tool_result.is_success:
                consecutive_failures += 1
                logger.warning(
                    f"[Agent] Tool '{tool_name}' failed "
                    f"({consecutive_failures}/{self.state.max_consecutive_failures}): {content}"
                )
                if consecutive_failures >= self.state.max_consecutive_failures:
                    logger.error(
                        f"[Agent] Aborting: {self.state.max_consecutive_failures} "
                        f"consecutive tool failures reached."
                    )
                    return AgentResult(
                        is_done=False,
                        error=(
                            f"Agent aborted after {self.state.max_consecutive_failures} "
                            f"consecutive tool failures. Last error: {content}"
                        ),
                    )
            else:
                consecutive_failures = 0

            if tool_name == "done_tool":
                logger.info(f"[Agent] 📝 Final Answer: {content}")
                return self.result(content=content, is_done=True)

            if tool_name != "done_tool":
                logger.info(f"[Agent] 📝 Tool Response: {content}")

        logger.warning(
            f"[Agent] Max steps ({self.state.max_steps}) reached without completing the task."
        )
        return AgentResult(
            is_done=False,
            error=f"Agent reached the maximum number of steps ({self.state.max_steps}) without completing.",
        )

    def invoke(self, query: str) -> AgentResult:
        self.state.task = query
        # Create a per-run file logger for the chain of thought
        file_handler = _create_file_logger() if self.log_to_file else None
        if file_handler:
            logger.addHandler(file_handler)
        logger.info(f"[Agent] 🔍 Query: {query}")
        try:
            with self.desktop.auto_minimize() if self.auto_minimize else nullcontext():
                self.watchdog.set_focus_callback(self.desktop.tree.on_focus_changed)
                with self.watchdog:
                    result = self.loop()
            if result.is_done:
                logger.info(f"[Agent] Task completed successfully")
            else:
                logger.warning(f"[Agent] Task ended with error: {result.error}")
            return result
        except Exception as e:
            logger.error(f"[Agent] Unhandled exception: {e}", exc_info=True)
            raise
        finally:
            if file_handler:
                logger.removeHandler(file_handler)
                file_handler.close()