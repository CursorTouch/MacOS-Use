"""
Configuration constants for macOS accessibility tree traversal.
Re-exports from the centralized ax module for backward compatibility.

All constants are now defined in macos_use.ax.enums and re-exported here
so existing code that imports from tree.config continues to work.
"""

from macos_use.ax.enums import (
    INTERACTIVE_ROLES,
    NON_INTERACTIVE_ROLES,
    INTERACTIVE_ACTIONS,
    WINDOW_CONTROL_SUBROLES,
    SCROLLABLE_ROLES,
    CONTAINER_ROLES,
    Action as Actions,
)
