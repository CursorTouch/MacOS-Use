"""
Data classes for representing macOS desktop state.
"""
from macos_use.agent.tree.views import TreeState, BoundingBox
from dataclasses import dataclass
from tabulate import tabulate
from typing import Optional
from PIL.Image import Image
from enum import Enum


class Browser(Enum):
    """Supported browser applications."""
    SAFARI = 'com.apple.Safari'
    CHROME = 'com.google.Chrome'
    FIREFOX = 'org.mozilla.firefox'
    EDGE = 'com.microsoft.edgemac'

    @classmethod
    def has_bundle_id(cls, bundle_id: str) -> bool:
        """Check if a bundle ID matches a known browser."""
        if not hasattr(cls, '_bundle_ids'):
            cls._bundle_ids = {b.value for b in cls}
        return bundle_id in cls._bundle_ids


class Status(Enum):
    """Window status enumeration."""
    NORMAL = 'Normal'
    MINIMIZED = 'Minimized'
    FULL_SCREEN = 'FullScreen'
    HIDDEN = 'Hidden'


@dataclass
class Window:
    """Represents a macOS window."""
    name: str
    is_browser: bool
    status: Status
    bounding_box: BoundingBox
    pid: int
    bundle_id: str = ''
    
    def to_row(self) -> list:
        """Convert to a table row for display."""
        return [
            self.name, 
            self.status.value, 
            int(self.bounding_box.width), 
            int(self.bounding_box.height), 
            self.bundle_id
        ]


@dataclass
class Size:
    """Represents screen dimensions."""
    width: int
    height: int

    def to_string(self) -> str:
        return f'({self.width},{self.height})'


@dataclass
class DesktopState:
    """Represents the complete state of the macOS desktop."""
    active_window: Optional[Window]
    windows: list[Window]
    screenshot: Optional[Image] = None
    tree_state: Optional[TreeState] = None

    def active_window_to_string(self) -> str:
        """Convert active window to table string."""
        if not self.active_window:
            return 'No active window found'
        headers = ["Name", "Status", "Width", "Height", "Bundle ID"]
        return tabulate([self.active_window.to_row()], headers=headers, tablefmt="simple")

    def windows_to_string(self) -> str:
        """Convert all windows to table string."""
        if not self.windows:
            return 'No windows found'
        headers = ["Name", "Status", "Width", "Height", "Bundle ID"]
        rows = [window.to_row() for window in self.windows]
        return tabulate(rows, headers=headers, tablefmt="simple")
