from macos_use.tree.views import BoundingBox, TreeState
from dataclasses import dataclass
from PIL.Image import Image
from typing import Union, Optional
from enum import Enum


class Status(Enum):
    ACTIVE = 'Active'
    FULLSCREEN = 'Fullscreen'
    VISIBLE = 'Visible'
    HIDDEN = 'Hidden'
    MINIMIZED = 'Minimized'
    WINDOWLESS = 'Windowless'


@dataclass
class Size:
    width: int
    height: int

    def to_string(self):
        return f'({self.width},{self.height})'


@dataclass
class Window:
    name: str
    is_browser: bool
    status: Status
    bounding_box: BoundingBox
    pid: int
    bundle_id: str


@dataclass
class DesktopState:
    active_window: Optional[Window]
    windows: list
    screenshot: Union[Image, bytes, None] = None
    tree_state: Optional[TreeState] = None

    def windows_to_string(self) -> str:
        """Format windows list for display."""
        if not self.windows:
            return "No open applications."
        lines = [f"{w.name} ({w.bundle_id}) - {w.status.value}" for w in self.windows]
        return "\n".join(lines)

    def active_window_to_string(self) -> str:
        """Format active window for display."""
        if self.active_window is None:
            return "No focused window."
        w = self.active_window
        return f"{w.name} ({w.bundle_id}) - {w.status.value}"
