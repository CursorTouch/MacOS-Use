from macos_use.agent.desktop.views import DesktopState, Browser
from macos_use.agent.registry.views import ToolResult
from macos_use.agent.desktop.service import Desktop
from importlib.resources import files
from datetime import datetime
from getpass import getuser
from typing import Literal
from pathlib import Path
import macos_use.ax as ax

class Prompt:
    @staticmethod
    def system(mode:Literal["flash","normal"],desktop:Desktop,browser: Browser,max_steps:int,instructions: list[str]=[]) -> str:
        width, height = ax.GetScreenSize()
        match mode:
            case "flash":
                template =Path(files('macos_use.agent.prompt').joinpath('system_flash.md')).read_text(encoding='utf-8')
                return template.format(**{
                    'max_steps': max_steps,
                    'datetime': datetime.now().strftime('%A, %B %d, %Y'),
                    'os':desktop.get_macos_version(),
                    'browser':browser.value,
                })
            case "normal":
                template =Path(files('macos_use.agent.prompt').joinpath('system.md')).read_text(encoding='utf-8')
                return template.format(**{
                    'datetime': datetime.now().strftime('%A, %B %d, %Y'),
                    'instructions': '\n'.join(instructions),
                    'download_directory': Path.home().joinpath('Downloads').as_posix(),
                    'os':desktop.get_macos_version(),
                    'language':desktop.get_default_language(),
                    'browser':browser.value,
                    'home_dir':Path.home().as_posix(),
                    'user':f"{getuser()} ({desktop.get_user_account_type()})",
                    'resolution':f'Primary Monitor ({width}x{height}) with DPI Scale: {desktop.get_dpi_scaling()}',
                    'max_steps': max_steps
                })
            case _:
                raise ValueError(f"Invalid mode: {mode} (must be 'flash' or 'normal')")
         
    @staticmethod
    def human(query:str,step:int,max_steps:int,desktop:Desktop) -> str:
        cursor_location = ax.GetCursorPos()
        desktop_state=desktop.desktop_state
        template = Path(files('macos_use.agent.prompt').joinpath('human.md')).read_text(encoding='utf-8')

        return template.format(**{
            'steps': step,
            'max_steps': max_steps,
            'active_window': desktop_state.active_window_to_string(),
            'windows': desktop_state.windows_to_string(),
            'cursor_location': f'({cursor_location[0]},{cursor_location[1]})',
            'interactive_elements': desktop_state.tree_state.interactive_elements_to_string() if desktop.use_accessibility else 'No accessability data is available',
            'scrollable_elements': desktop_state.tree_state.scrollable_elements_to_string() if desktop.use_accessibility else 'No accessability data is available',
            'query':query
        })

    
