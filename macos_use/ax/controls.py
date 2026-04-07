"""
Control classes wrapping macOS AXUIElementRef.
Provides a Pythonic, object-oriented interface to accessibility elements
with typed subclasses for specific control types.

Equivalent to the Windows UIA controls.py module, adapted for macOS.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, List, Callable, Any, Union

from ApplicationServices import (
    AXUIElementCreateApplication,
    AXUIElementCreateSystemWide,
    AXUIElementCopyAttributeValue,
    AXUIElementSetAttributeValue,
    AXUIElementPerformAction,
    AXUIElementCopyAttributeNames,
    AXUIElementCopyActionNames,
    AXUIElementIsAttributeSettable,
    AXUIElementGetAttributeValueCount,
    kAXErrorSuccess,
)

from .enums import (
    Role,
    Subrole,
    Attribute,
    Action,
    AXError,
    INTERACTIVE_ROLES,
    INTERACTIVE_ACTIONS,
    SCROLLABLE_ROLES,
    CONTAINER_ROLES,
    WINDOW_CONTROL_SUBROLES,
    ActivationPolicyNames,
)
from .core import (
    Rect,
    Point,
    Size,
    GetAttribute,
    SetAttribute,
    GetAttributeNames,
    GetActionNames,
    PerformAction,
    GetChildren as _GetChildren,
    GetPosition,
    GetSize,
    GetRect,
    GetChildCount,
    IsAttributeSettable,
    GetElementPid,
    GetForegroundWindowPID,
    Click as _Click,
    RightClick as _RightClick,
    DoubleClick as _DoubleClick,
    MiddleClick as _MiddleClick,
    TypeText as _TypeText,
    WheelDown as _WheelDown,
    WheelUp as _WheelUp,
    DragTo as _DragTo,
    MoveTo as _MoveTo,
)

logger = logging.getLogger(__name__)


class Control:
    """
    Base Control class wrapping a macOS AXUIElementRef.
    Equivalent to Windows UIA Control class.

    Provides property-based access to accessibility attributes,
    child navigation, element search, and action execution.
    """

    def __init__(self, element=None, pid: Optional[int] = None):
        """
        Create a Control from an AXUIElement or application PID.

        Args:
            element: An AXUIElementRef to wrap.
            pid: Process ID to create an application element from.
        """
        if element is not None:
            self._element = element
        elif pid is not None:
            self._element = AXUIElementCreateApplication(pid)
        else:
            self._element = AXUIElementCreateSystemWide()

    @property
    def Element(self):
        """Get the underlying AXUIElementRef."""
        return self._element

    # =========================================================================
    # Standard Properties
    # =========================================================================

    @property
    def Role(self) -> str:
        """Get the role of this element (e.g., 'AXButton', 'AXTextField')."""
        return GetAttribute(self._element, Attribute.Role) or ''

    @property
    def Subrole(self) -> str:
        """Get the subrole of this element (e.g., 'AXCloseButton')."""
        return GetAttribute(self._element, Attribute.Subrole) or ''

    @property
    def RoleDescription(self) -> str:
        """Get the localized role description (e.g., 'button', 'text field')."""
        return GetAttribute(self._element, Attribute.RoleDescription) or ''

    @property
    def Title(self) -> str:
        """Get the title/label of this element."""
        return GetAttribute(self._element, Attribute.Title) or ''

    @property
    def Name(self) -> str:
        """
        Get the display name of this element.
        Tries Title, then Description, then Value.
        """
        return self.Title or self.Description or ''

    @property
    def Description(self) -> str:
        """Get the accessibility description of this element."""
        return GetAttribute(self._element, Attribute.Description) or ''

    @property
    def Help(self) -> str:
        """Get the help text of this element."""
        return GetAttribute(self._element, Attribute.Help) or ''

    @property
    def Value(self):
        """Get the value of this element (type varies by element)."""
        return GetAttribute(self._element, Attribute.Value)

    @Value.setter
    def Value(self, value) -> None:
        """Set the value of this element."""
        SetAttribute(self._element, Attribute.Value, value)

    @property
    def ValueString(self) -> str:
        """Get the value as a string."""
        val = self.Value
        return str(val) if val is not None else ''

    @property
    def Identifier(self) -> str:
        """Get the unique identifier (similar to Windows AutomationId)."""
        return GetAttribute(self._element, Attribute.Identifier) or ''

    @property
    def IsEnabled(self) -> bool:
        """Check if this element is enabled."""
        val = GetAttribute(self._element, Attribute.Enabled)
        return val is not False

    @property
    def IsFocused(self) -> bool:
        """Check if this element has keyboard focus."""
        val = GetAttribute(self._element, Attribute.Focused)
        return val is True

    @IsFocused.setter
    def IsFocused(self, value: bool) -> None:
        """Set focus on this element."""
        SetAttribute(self._element, Attribute.Focused, value)

    @property
    def IsHidden(self) -> bool:
        """Check if this element is hidden."""
        val = GetAttribute(self._element, Attribute.Hidden)
        return val is True

    @property
    def IsSelected(self) -> bool:
        """Check if this element is selected."""
        val = GetAttribute(self._element, Attribute.Selected)
        return val is True

    # =========================================================================
    # Geometry Properties
    # =========================================================================

    @property
    def Position(self) -> Optional[Point]:
        """Get the position of this element in screen coordinates."""
        pos = GetPosition(self._element)
        if pos:
            return Point(x=pos[0], y=pos[1])
        return None

    @property
    def ElementSize(self) -> Optional[Size]:
        """Get the size of this element."""
        size = GetSize(self._element)
        if size:
            return Size(width=size[0], height=size[1])
        return None

    @property
    def BoundingRectangle(self) -> Optional[Rect]:
        """
        Get the bounding rectangle of this element.
        Equivalent to Windows UIA BoundingRectangle property.
        """
        return GetRect(self._element)

    @property
    def Center(self) -> Optional[Point]:
        """Get the center point of this element."""
        rect = self.BoundingRectangle
        if rect:
            cx, cy = rect.center
            return Point(x=cx, y=cy)
        return None

    # =========================================================================
    # Window Properties
    # =========================================================================

    @property
    def IsMain(self) -> bool:
        """Check if this window is the main window."""
        val = GetAttribute(self._element, Attribute.Main)
        return val is True

    @property
    def IsMinimized(self) -> bool:
        """Check if this window is minimized."""
        val = GetAttribute(self._element, Attribute.Minimized)
        return val is True

    @property
    def IsFullScreen(self) -> bool:
        """Check if this window is in fullscreen mode."""
        val = GetAttribute(self._element, Attribute.FullScreen)
        if val is True:
            return True
        # Fallback check: subrole
        return self.Subrole == Subrole.FullScreenWindow

    @property
    def IsModal(self) -> bool:
        """Check if this window is modal."""
        val = GetAttribute(self._element, Attribute.Modal)
        return val is True

    # =========================================================================
    # Navigation Properties
    # =========================================================================

    @property
    def Parent(self) -> Optional['Control']:
        """Get the parent element."""
        parent = GetAttribute(self._element, Attribute.Parent)
        if parent:
            return Control(element=parent)
        return None

    @property
    def Window(self) -> Optional['Control']:
        """Get the containing window element."""
        window = GetAttribute(self._element, Attribute.Window)
        if window:
            return Control(element=window)
        return None

    @property
    def TopLevelUIElement(self) -> Optional['Control']:
        """Get the top-level UI element."""
        top = GetAttribute(self._element, Attribute.TopLevelUIElement)
        if top:
            return Control(element=top)
        return None

    @property
    def ChildCount(self) -> int:
        """Get the number of child elements."""
        return GetChildCount(self._element)

    # =========================================================================
    # Child Access Methods
    # =========================================================================

    def GetChildren(self) -> List['Control']:
        """
        Get all child controls.
        Equivalent to Windows UIA GetChildren().
        """
        children = _GetChildren(self._element)
        return [Control(element=child) for child in children]

    def GetFirstChildControl(self) -> Optional['Control']:
        """Get the first child control."""
        children = _GetChildren(self._element)
        if children:
            return Control(element=children[0])
        return None

    def GetLastChildControl(self) -> Optional['Control']:
        """Get the last child control."""
        children = _GetChildren(self._element)
        if children:
            return Control(element=children[-1])
        return None

    # =========================================================================
    # Application-specific access
    # =========================================================================

    @property
    def FocusedWindow(self) -> Optional['Control']:
        """Get the focused window of this application element."""
        window = GetAttribute(self._element, Attribute.FocusedWindow)
        if window:
            return Control(element=window)
        return None

    @property
    def MainWindow(self) -> Optional['Control']:
        """Get the main window of this application element."""
        window = GetAttribute(self._element, Attribute.MainWindow)
        if window:
            return Control(element=window)
        return None

    @property
    def Windows(self) -> List['Control']:
        """Get all windows of this application element."""
        windows = GetAttribute(self._element, Attribute.Windows)
        if windows:
            return [Control(element=w) for w in windows]
        return []

    @property
    def MenuBar(self) -> Optional['Control']:
        """Get the menu bar of this application element."""
        menu_bar = GetAttribute(self._element, Attribute.MenuBar)
        if menu_bar:
            return Control(element=menu_bar)
        return None

    @property
    def ExtrasMenuBar(self) -> Optional['Control']:
        """Get the extras menu bar (status bar items) of this application element."""
        extras = GetAttribute(self._element, Attribute.ExtrasMenuBar)
        if extras:
            return Control(element=extras)
        return None

    # =========================================================================
    # Scroll-specific Properties
    # =========================================================================

    @property
    def HorizontalScrollBar(self) -> Optional['Control']:
        """Get the horizontal scroll bar."""
        sb = GetAttribute(self._element, Attribute.HorizontalScrollBar)
        if sb:
            return Control(element=sb)
        return None

    @property
    def VerticalScrollBar(self) -> Optional['Control']:
        """Get the vertical scroll bar."""
        sb = GetAttribute(self._element, Attribute.VerticalScrollBar)
        if sb:
            return Control(element=sb)
        return None

    @property
    def IsHorizontallyScrollable(self) -> bool:
        """Check if the element can scroll horizontally."""
        return self.HorizontalScrollBar is not None

    @property
    def IsVerticallyScrollable(self) -> bool:
        """Check if the element can scroll vertically."""
        return self.VerticalScrollBar is not None

    # =========================================================================
    # Table/Grid Properties
    # =========================================================================

    @property
    def Rows(self) -> List['Control']:
        """Get all rows (for tables/outlines)."""
        rows = GetAttribute(self._element, Attribute.Rows)
        if rows:
            return [Control(element=r) for r in rows]
        return []

    @property
    def VisibleRows(self) -> List['Control']:
        """Get visible rows."""
        rows = GetAttribute(self._element, Attribute.VisibleRows)
        if rows:
            return [Control(element=r) for r in rows]
        return []

    @property
    def Columns(self) -> List['Control']:
        """Get all columns."""
        cols = GetAttribute(self._element, Attribute.Columns)
        if cols:
            return [Control(element=c) for c in cols]
        return []

    @property
    def SelectedRows(self) -> List['Control']:
        """Get selected rows."""
        rows = GetAttribute(self._element, Attribute.SelectedRows)
        if rows:
            return [Control(element=r) for r in rows]
        return []

    # =========================================================================
    # Expand/Collapse Properties
    # =========================================================================

    @property
    def IsExpanded(self) -> bool:
        """Check if this element is expanded."""
        val = GetAttribute(self._element, Attribute.Expanded)
        return val is True

    # =========================================================================
    # Text Properties
    # =========================================================================

    @property
    def NumberOfCharacters(self) -> int:
        """Get the number of characters in a text element."""
        val = GetAttribute(self._element, Attribute.NumberOfCharacters)
        return val if val is not None else 0

    @property
    def SelectedText(self) -> str:
        """Get the selected text."""
        return GetAttribute(self._element, Attribute.SelectedText) or ''

    @property
    def SelectedTextRange(self):
        """Get the selected text range."""
        return GetAttribute(self._element, Attribute.SelectedTextRange)

    @property
    def VisibleCharacterRange(self):
        """Get the visible character range."""
        return GetAttribute(self._element, Attribute.VisibleCharacterRange)

    @property
    def PlaceholderValue(self) -> str:
        """Get placeholder text (for text fields)."""
        return GetAttribute(self._element, Attribute.PlaceholderValue) or ''

    # =========================================================================
    # Misc Properties
    # =========================================================================

    @property
    def URL(self) -> str:
        """Get the URL (for links and web elements)."""
        val = GetAttribute(self._element, Attribute.URL)
        return str(val) if val else ''

    @property
    def Document(self) -> str:
        """Get the document path/URL."""
        val = GetAttribute(self._element, Attribute.Document)
        return str(val) if val else ''

    # =========================================================================
    # Attribute Inspection
    # =========================================================================

    @property
    def AttributeNames(self) -> list:
        """Get all attribute names supported by this element."""
        return GetAttributeNames(self._element)

    @property
    def ActionNames(self) -> list:
        """Get all action names supported by this element."""
        return GetActionNames(self._element)

    def GetAttributeValue(self, attribute: str):
        """Get a specific attribute value by name."""
        return GetAttribute(self._element, attribute)

    def SetAttributeValue(self, attribute: str, value) -> bool:
        """Set a specific attribute value by name."""
        return SetAttribute(self._element, attribute, value)

    def IsAttributeSettable(self, attribute: str) -> bool:
        """Check if an attribute can be set."""
        return IsAttributeSettable(self._element, attribute)

    # =========================================================================
    # Actions
    # =========================================================================

    def PerformAction(self, action: str) -> bool:
        """
        Perform an action on this element.
        Returns True if successful.
        """
        return PerformAction(self._element, action)

    def Press(self) -> bool:
        """Perform AXPress action (click/activate)."""
        return self.PerformAction(Action.Press)

    def Confirm(self) -> bool:
        """Perform AXConfirm action."""
        return self.PerformAction(Action.Confirm)

    def Cancel(self) -> bool:
        """Perform AXCancel action."""
        return self.PerformAction(Action.Cancel)

    def Increment(self) -> bool:
        """Perform AXIncrement action."""
        return self.PerformAction(Action.Increment)

    def Decrement(self) -> bool:
        """Perform AXDecrement action."""
        return self.PerformAction(Action.Decrement)

    def ShowMenu(self) -> bool:
        """Perform AXShowMenu action (right-click menu)."""
        return self.PerformAction(Action.ShowMenu)

    def Pick(self) -> bool:
        """Perform AXPick action."""
        return self.PerformAction(Action.Pick)

    def Raise(self) -> bool:
        """Perform AXRaise action (bring to front)."""
        return self.PerformAction(Action.Raise)

    def SetFocus(self) -> bool:
        """Set focus to this element."""
        return SetAttribute(self._element, Attribute.Focused, True)

    # =========================================================================
    # Search Methods
    # =========================================================================

    def FindAll(
        self,
        role: Optional[str] = None,
        subrole: Optional[str] = None,
        title: Optional[str] = None,
        identifier: Optional[str] = None,
        predicate: Optional[Callable[['Control'], bool]] = None,
        max_depth: int = 25,
    ) -> List['Control']:
        """
        Find all descendant controls matching the given criteria.
        Equivalent to Windows UIA FindAll().

        Args:
            role: Filter by AXRole.
            subrole: Filter by AXSubrole.
            title: Filter by AXTitle (substring match).
            identifier: Filter by AXIdentifier.
            predicate: Custom filter function.
            max_depth: Maximum search depth.

        Returns:
            List of matching Control objects.
        """
        results = []
        self._find_recursive(
            self._element, results, role, subrole, title,
            identifier, predicate, max_depth, 0, find_first=False
        )
        return results

    def FindFirst(
        self,
        role: Optional[str] = None,
        subrole: Optional[str] = None,
        title: Optional[str] = None,
        identifier: Optional[str] = None,
        predicate: Optional[Callable[['Control'], bool]] = None,
        max_depth: int = 25,
    ) -> Optional['Control']:
        """
        Find the first descendant control matching the given criteria.
        Equivalent to Windows UIA FindFirst().
        """
        results = []
        self._find_recursive(
            self._element, results, role, subrole, title,
            identifier, predicate, max_depth, 0, find_first=True
        )
        return results[0] if results else None

    def _find_recursive(
        self,
        element,
        results: list,
        role: Optional[str],
        subrole: Optional[str],
        title: Optional[str],
        identifier: Optional[str],
        predicate: Optional[Callable],
        max_depth: int,
        current_depth: int,
        find_first: bool,
    ) -> None:
        """Recursive element search helper."""
        if current_depth > max_depth:
            return
        if find_first and results:
            return

        children = _GetChildren(element)
        for child in children:
            if find_first and results:
                return

            control = Control(element=child)

            # Apply filters
            match = True
            if role and control.Role != role:
                match = False
            if match and subrole and control.Subrole != subrole:
                match = False
            if match and title and title not in control.Title:
                match = False
            if match and identifier and control.Identifier != identifier:
                match = False
            if match and predicate and not predicate(control):
                match = False

            if match and (role or subrole or title or identifier or predicate):
                results.append(control)
                if find_first:
                    return

            # Recurse into children
            self._find_recursive(
                child, results, role, subrole, title,
                identifier, predicate, max_depth, current_depth + 1, find_first
            )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @property
    def IsInteractive(self) -> bool:
        """Check if this element is interactive (clickable, editable, etc.)."""
        role = self.Role
        if role in INTERACTIVE_ROLES:
            return True
        actions = self.ActionNames
        if actions:
            return any(a in INTERACTIVE_ACTIONS for a in actions)
        return self.Subrole in WINDOW_CONTROL_SUBROLES

    @property
    def IsContainer(self) -> bool:
        """Check if this element is a container."""
        role = self.Role
        if role in CONTAINER_ROLES:
            # Exception: AXGroup with actions and label is interactive
            if role == Role.Group:
                actions = self.ActionNames
                has_actions = actions and any(a in INTERACTIVE_ACTIONS for a in actions)
                has_label = bool(self.Title or self.Description or self.ValueString)
                if has_actions and has_label:
                    return False
            return True
        return False

    @property
    def IsScrollable(self) -> bool:
        """Check if this element is scrollable."""
        return self.Role in SCROLLABLE_ROLES

    def HasAction(self, action: str) -> bool:
        """Check if this element supports a specific action."""
        return action in self.ActionNames

    @property
    def Label(self) -> str:
        """
        Get a human-readable label for this element.
        Tries Title, Description, Value, then subrole friendly name.
        """
        title = self.Title
        if title:
            return title
        desc = self.Description
        if desc:
            return desc
        val = self.Value
        if val is not None and isinstance(val, str) and val:
            return val
        subrole = self.Subrole
        if subrole in WINDOW_CONTROL_SUBROLES:
            return WINDOW_CONTROL_SUBROLES[subrole]
        return "(no label)"

    # =========================================================================
    # Representation
    # =========================================================================

    def __str__(self) -> str:
        role = self.Role
        name = self.Label
        rect = self.BoundingRectangle
        return f'Control(Role={role}, Name={name!r}, Rect={rect})'

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} Role={self.Role!r} Name={self.Label!r}>'

    def __eq__(self, other) -> bool:
        if isinstance(other, Control):
            return self._element == other._element
        return NotImplemented

    def __hash__(self):
        return hash(id(self._element))


# =============================================================================
# Typed Control Subclasses
# =============================================================================

class ApplicationControl(Control):
    """
    Control for AXApplication elements.

    Wraps both the AXUIElementRef (for accessibility tree access) and
    the NSRunningApplication (for process-level metadata like bundle ID,
    icon, launch date, hidden state, etc.).

    The NSRunningApplication is lazily resolved from the PID on first access.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ns_running_app = None  # Lazily resolved

    def _get_ns_running_app(self):
        """Lazily resolve the NSRunningApplication for this application's PID."""
        if self._ns_running_app is None:
            pid = GetElementPid(self.Element)
            if pid is not None:
                from Cocoa import NSWorkspace
                for app in NSWorkspace.sharedWorkspace().runningApplications():
                    if app.processIdentifier() == pid:
                        self._ns_running_app = app
                        break
        return self._ns_running_app

    def __str__(self) -> str:
        return (
            f"App(Name={self.Name!r}, Status={self.Status!r}, "
            f"Policy={self.ActivationPolicy!r})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def Name(self) -> str:
        """Get the display name of this application."""
        name = self.Title
        return name if name else self.LocalizedName

    @property
    def IsMinimized(self) -> bool:
        """Check if all of this application's windows are minimized."""
        windows = self.Windows
        if not windows:
            return False
        return all(w.IsMinimized for w in windows)

    @property
    def IsFullScreen(self) -> bool:
        """Check if any of this application's windows are in fullscreen mode."""
        windows = self.Windows
        if not windows:
            return False
        return any(w.IsFullScreen for w in windows)

    @property
    def Status(self) -> str:
        """
        Get a human-readable status summarizing the application's current state.

        Returns one of: 'Active', 'Fullscreen', 'Visible', 'Hidden', 'Minimized', 'Windowless'.
        """
        if self.IsHidden:
            return 'Hidden'
        windows = self.Windows
        if not windows:
            return 'Windowless'
        if all(w.IsMinimized for w in windows):
            return 'Minimized'
        if self.IsActive:
            if any(w.IsFullScreen for w in windows):
                return 'Fullscreen'
            return 'Active'
        return 'Visible'

    @property
    def FocusedUIElement(self) -> Optional[Control]:
        """Get the currently focused UI element in this application."""
        elem = GetAttribute(self.Element, Attribute.FocusedUIElement)
        if elem:
            return CreateControl(elem)
        return None

    @property
    def FocusedWindow(self) -> Optional[WindowControl]:
        """Get the focused window of this application element."""
        window = GetAttribute(self.Element, Attribute.FocusedWindow)
        if window:
            return CreateControl(window)
        return None

    @property
    def MainWindow(self) -> Optional[WindowControl]:
        """Get the main window of this application element."""
        window = GetAttribute(self.Element, Attribute.MainWindow)
        if window:
            return CreateControl(window)
        return None

    @property
    def Windows(self) -> List[WindowControl]:
        """Get all windows of this application element."""
        windows = GetAttribute(self.Element, Attribute.Windows)
        if windows:
            return [CreateControl(w) for w in windows]
        return []

    @property
    def IsApplicationRunning(self) -> bool:
        """Check if the application is running (AX attribute)."""
        val = GetAttribute(self.Element, Attribute.IsApplicationRunning)
        return val is True

    @property
    def EnhancedUserInterface(self) -> bool:
        """Get the enhanced user interface flag."""
        val = GetAttribute(self.Element, Attribute.Enhanced)
        return val is True

    @EnhancedUserInterface.setter
    def EnhancedUserInterface(self, value: bool) -> None:
        """Set the enhanced user interface flag (enables deeper tree access in some apps)."""
        SetAttribute(self.Element, Attribute.Enhanced, value)

    @property
    def PID(self) -> Optional[int]:
        """Get the process identifier (PID) of this application."""
        return GetElementPid(self.Element)

    @property
    def BundleIdentifier(self) -> Optional[str]:
        """Get the CFBundleIdentifier of this application."""
        app = self._get_ns_running_app()
        if app:
            val = app.bundleIdentifier()
            return str(val) if val else None
        return None

    @property
    def BundleURL(self) -> Optional[str]:
        """Get the file URL to the application's .app bundle."""
        app = self._get_ns_running_app()
        if app:
            url = app.bundleURL()
            return str(url) if url else None
        return None

    @property
    def ExecutableURL(self) -> Optional[str]:
        """Get the file URL to the application's executable binary."""
        app = self._get_ns_running_app()
        if app:
            url = app.executableURL()
            return str(url) if url else None
        return None

    @property
    def LocalizedName(self) -> Optional[str]:
        """Get the localized display name of the application."""
        app = self._get_ns_running_app()
        if app:
            val = app.localizedName()
            return str(val) if val else None
        return None

    @property
    def Icon(self):
        """Get the application icon as an NSImage."""
        app = self._get_ns_running_app()
        if app:
            return app.icon()
        return None

    @property
    def LaunchDate(self):
        """Get the date and time when the application was launched."""
        app = self._get_ns_running_app()
        if app:
            return app.launchDate()
        return None

    @property
    def IsActive(self) -> bool:
        """
        Check if this application is the currently active (frontmost) application.

        Uses CGWindowListCopyWindowInfo instead of NSRunningApplication.isActive()
        because the latter relies on NSRunLoop which may be stale in scripts.
        """
        pid = self.PID
        if pid is None:
            return False
        frontmost_pid = GetForegroundWindowPID()
        return pid == frontmost_pid

    @property
    def IsHidden(self) -> bool:
        """Check if this application is currently hidden."""
        app = self._get_ns_running_app()
        if app:
            return bool(app.isHidden())
        return False

    @property
    def IsFinishedLaunching(self) -> bool:
        """Check if the application has finished launching."""
        app = self._get_ns_running_app()
        if app:
            return bool(app.isFinishedLaunching())
        return False

    @property
    def IsTerminated(self) -> bool:
        """Check if the application has been terminated."""
        app = self._get_ns_running_app()
        if app:
            return bool(app.isTerminated())
        return False

    @property
    def ActivationPolicy(self) -> Optional[str]:
        """
        Get the activation policy of the application as a human-readable string.

        Returns: 'Regular', 'Accessory', 'Prohibited', or None.
        """
        app = self._get_ns_running_app()
        if app:
            policy = int(app.activationPolicy())
            return ActivationPolicyNames.get(policy, f'Unknown({policy})')
        return None


class WindowControl(Control):
    """Control for AXWindow elements."""

    def Close(self) -> bool:
        """Close this window via the close button."""
        close_btn = GetAttribute(self._element, Attribute.CloseButton)
        if close_btn:
            return PerformAction(close_btn, Action.Press)
        return False

    def Minimize(self) -> bool:
        """Minimize this window."""
        return SetAttribute(self._element, Attribute.Minimized, True)

    def Unminimize(self) -> bool:
        """Restore this minimized window."""
        return SetAttribute(self._element, Attribute.Minimized, False)

    def Zoom(self) -> bool:
        """Zoom (maximize) this window via the zoom button."""
        zoom_btn = GetAttribute(self._element, Attribute.ZoomButton)
        if zoom_btn:
            return PerformAction(zoom_btn, Action.Press)
        return False

    def Resize(self, width: float, height: float) -> bool:
        """Resize this window to the specified dimensions."""
        return SetAttribute(self._element, Attribute.Size, (width, height))

    def MoveWindowTo(self, x: float, y: float) -> bool:
        """Move this window to the specified screen position."""
        return SetAttribute(self._element, Attribute.Position, (x, y))

    @property
    def DefaultButton(self) -> Optional[Control]:
        """Get the default button of this window/dialog."""
        btn = GetAttribute(self._element, Attribute.DefaultButton)
        if btn:
            return Control(element=btn)
        return None

    @property
    def CancelButton(self) -> Optional[Control]:
        """Get the cancel button of this window/dialog."""
        btn = GetAttribute(self._element, Attribute.CancelButton)
        if btn:
            return Control(element=btn)
        return None


class ButtonControl(Control):
    """Control for AXButton elements."""

    def Click(self) -> bool:
        """Click this button."""
        return self.Press()


class CheckBoxControl(Control):
    """Control for AXCheckBox elements."""

    @property
    def IsChecked(self) -> bool:
        """Check if the checkbox is checked."""
        val = self.Value
        return val == 1 or val is True

    def Toggle(self) -> bool:
        """Toggle the checkbox."""
        return self.Press()


class RadioButtonControl(Control):
    """Control for AXRadioButton elements."""

    @property
    def IsSelected(self) -> bool:
        """Check if this radio button is selected."""
        val = self.Value
        return val == 1 or val is True

    def Select(self) -> bool:
        """Select this radio button."""
        return self.Press()


class TextFieldControl(Control):
    """Control for AXTextField elements."""

    def SetText(self, text: str) -> bool:
        """Set the text value of this field."""
        return SetAttribute(self._element, Attribute.Value, text)

    def GetText(self) -> str:
        """Get the text value of this field."""
        return self.ValueString

    @property
    def InsertionPoint(self) -> int:
        """Get the insertion point line number."""
        val = GetAttribute(self._element, Attribute.InsertionPointLineNumber)
        return val if val is not None else 0

    def ClearText(self) -> bool:
        """Clear the text in this field."""
        return self.SetText('')


class TextAreaControl(TextFieldControl):
    """Control for AXTextArea elements (multi-line text)."""
    pass


class ComboBoxControl(Control):
    """Control for AXComboBox elements."""

    def SetText(self, text: str) -> bool:
        """Set the text value."""
        return SetAttribute(self._element, Attribute.Value, text)

    def Expand(self) -> bool:
        """Expand the dropdown."""
        return self.Press()


class PopUpButtonControl(Control):
    """Control for AXPopUpButton elements (dropdown menus)."""

    def Open(self) -> bool:
        """Open the pop-up menu."""
        return self.Press()


class SliderControl(Control):
    """Control for AXSlider elements."""

    @property
    def SliderValue(self) -> float:
        """Get the slider value as a float."""
        val = self.Value
        try:
            return float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    @SliderValue.setter
    def SliderValue(self, value: float) -> None:
        """Set the slider value."""
        SetAttribute(self._element, Attribute.Value, value)

    @property
    def MinValue(self) -> float:
        """Get the minimum value."""
        val = GetAttribute(self._element, Attribute.MinValue)
        return float(val) if val is not None else 0.0

    @property
    def MaxValue(self) -> float:
        """Get the maximum value."""
        val = GetAttribute(self._element, Attribute.MaxValue)
        return float(val) if val is not None else 100.0


class MenuItemControl(Control):
    """Control for AXMenuItem elements."""

    @property
    def MenuItemCmdChar(self) -> str:
        """Get the command character shortcut."""
        return GetAttribute(self._element, Attribute.MenuItemCmdChar) or ''

    def Select(self) -> bool:
        """Select this menu item."""
        return self.Press()


class MenuBarItemControl(Control):
    """Control for AXMenuBarItem elements."""

    def Open(self) -> bool:
        """Open this menu bar item."""
        return self.Press()


class TabControl(Control):
    """Control for AXTab elements."""

    def Select(self) -> bool:
        """Select this tab."""
        return self.Press()


class ListControl(Control):
    """Control for AXList elements."""

    @property
    def SelectedChildren(self) -> List[Control]:
        """Get selected children."""
        children = GetAttribute(self._element, Attribute.SelectedChildren)
        if children:
            return [Control(element=c) for c in children]
        return []


class TableControl(Control):
    """Control for AXTable elements."""

    @property
    def RowCount(self) -> int:
        """Get the number of rows."""
        val = GetAttribute(self._element, Attribute.RowCount)
        return val if val is not None else 0

    @property
    def ColumnCount(self) -> int:
        """Get the number of columns."""
        val = GetAttribute(self._element, Attribute.ColumnCount)
        return val if val is not None else 0

    @property
    def Header(self) -> Optional[Control]:
        """Get the table header."""
        header = GetAttribute(self._element, Attribute.Header)
        if header:
            return Control(element=header)
        return None


class OutlineControl(Control):
    """Control for AXOutline elements (tree views)."""

    @property
    def DisclosedRows(self) -> List[Control]:
        """Get disclosed (visible) rows."""
        rows = GetAttribute(self._element, Attribute.DisclosedRows)
        if rows:
            return [Control(element=r) for r in rows]
        return []


class ScrollAreaControl(Control):
    """Control for AXScrollArea elements."""

    @property
    def Contents(self) -> List[Control]:
        """Get the contents of the scroll area."""
        contents = GetAttribute(self._element, Attribute.Contents)
        if contents:
            return [Control(element=c) for c in contents]
        return []

    def GetScrollPosition(self) -> tuple:
        """
        Get the scroll position as (horizontal_percent, vertical_percent).
        Values are 0.0 to 1.0.
        """
        h_pct = 0.0
        v_pct = 0.0
        h_bar = GetAttribute(self._element, Attribute.HorizontalScrollBar)
        if h_bar:
            val = GetAttribute(h_bar, Attribute.Value)
            if val is not None:
                try:
                    h_pct = float(val)
                except (TypeError, ValueError):
                    pass
        v_bar = GetAttribute(self._element, Attribute.VerticalScrollBar)
        if v_bar:
            val = GetAttribute(v_bar, Attribute.Value)
            if val is not None:
                try:
                    v_pct = float(val)
                except (TypeError, ValueError):
                    pass
        return (h_pct, v_pct)


class GroupControl(Control):
    """Control for AXGroup elements."""
    pass


class ImageControl(Control):
    """Control for AXImage elements."""

    @property
    def URL(self) -> str:
        """Get the image URL if available."""
        val = GetAttribute(self._element, Attribute.URL)
        return str(val) if val else ''


class LinkControl(Control):
    """Control for AXLink elements."""

    @property
    def URL(self) -> str:
        """Get the link URL."""
        val = GetAttribute(self._element, Attribute.URL)
        return str(val) if val else ''


class ProgressIndicatorControl(Control):
    """Control for AXProgressIndicator elements."""

    @property
    def ProgressValue(self) -> float:
        """Get the progress value."""
        val = self.Value
        try:
            return float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            return 0.0


class StaticTextControl(Control):
    """Control for AXStaticText elements."""

    @property
    def Text(self) -> str:
        """Get the text content."""
        return self.ValueString or self.Title


class WebAreaControl(Control):
    """Control for AXWebArea elements."""

    @property
    def URL(self) -> str:
        """Get the web page URL."""
        val = GetAttribute(self._element, Attribute.URL)
        return str(val) if val else ''

    @property
    def Document(self) -> str:
        """Get the document URL."""
        val = GetAttribute(self._element, Attribute.Document)
        return str(val) if val else ''


class DisclosureTriangleControl(Control):
    """Control for AXDisclosureTriangle elements."""

    @property
    def IsExpanded(self) -> bool:
        """Check if the disclosure is expanded."""
        val = self.Value
        return val == 1 or val is True

    def Toggle(self) -> bool:
        """Toggle the disclosure state."""
        return self.Press()


class DockItemControl(Control):
    """Control for AXDockItem elements."""
    pass


class CellControl(Control):
    """Control for AXCell elements."""

    @property
    def RowIndex(self) -> int:
        """Get the row index of this cell."""
        val = GetAttribute(self._element, Attribute.Index)
        return val if val is not None else -1


class RowControl(Control):
    """Control for AXRow/AXOutlineRow elements."""

    @property
    def Index(self) -> int:
        """Get the row index."""
        val = GetAttribute(self._element, Attribute.Index)
        return val if val is not None else -1

    @property
    def DisclosureLevel(self) -> int:
        """Get the outline disclosure level."""
        val = GetAttribute(self._element, Attribute.DisclosureLevel)
        return val if val is not None else 0

    @property
    def IsDisclosed(self) -> bool:
        """Check if this outline row is disclosed (expanded)."""
        val = GetAttribute(self._element, Attribute.Expanded)
        return val is True


# =============================================================================
# Control Factory
# =============================================================================

# Role to typed Control class mapping
_ROLE_TO_CONTROL_CLASS = {
    Role.Application: ApplicationControl,
    Role.Window: WindowControl,
    Role.Button: ButtonControl,
    Role.CheckBox: CheckBoxControl,
    Role.RadioButton: RadioButtonControl,
    Role.TextField: TextFieldControl,
    Role.TextArea: TextAreaControl,
    Role.ComboBox: ComboBoxControl,
    Role.PopUpButton: PopUpButtonControl,
    Role.Slider: SliderControl,
    Role.MenuItem: MenuItemControl,
    Role.MenuBarItem: MenuBarItemControl,
    Role.Tab: TabControl,
    Role.List: ListControl,
    Role.Table: TableControl,
    Role.Outline: OutlineControl,
    Role.ScrollArea: ScrollAreaControl,
    Role.Group: GroupControl,
    Role.Image: ImageControl,
    Role.Link: LinkControl,
    Role.ProgressIndicator: ProgressIndicatorControl,
    Role.StaticText: StaticTextControl,
    Role.WebArea: WebAreaControl,
    Role.DisclosureTriangle: DisclosureTriangleControl,
    Role.DockItem: DockItemControl,
    Role.Cell: CellControl,
    Role.Row: RowControl,
    Role.OutlineRow: RowControl,
}


def CreateControl(element) -> Control:
    """
    Create the appropriate typed Control subclass for an AXUIElement.
    Equivalent to Windows UIA Control.CreateControlFromElement().

    Args:
        element: An AXUIElementRef.

    Returns:
        A typed Control subclass (e.g., ButtonControl, TextFieldControl).
    """
    role = GetAttribute(element, Attribute.Role)
    control_class = _ROLE_TO_CONTROL_CLASS.get(role, Control)
    return control_class(element=element)
