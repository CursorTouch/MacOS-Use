"""
macOS Accessibility (AX) module.
Provides a unified, Pythonic interface to the macOS Accessibility API.

This module is the macOS equivalent of the Windows UIA module.
It wraps the native AXUIElement framework into clean Python classes
with consistent patterns for element discovery, property access,
action execution, and event observation.

Structure:
    - enums: Constants (Roles, Subroles, Attributes, Actions, Notifications, KeyCodes)
    - core: Low-level functions (element creation, screen/mouse/keyboard, window management)
    - controls: Control classes wrapping AXUIElementRef (ButtonControl, TextFieldControl, etc.)
    - patterns: Interaction patterns (InvokePattern, ValuePattern, ScrollPattern, etc.)
    - events: Event observation (EventObserver, AppObserver)

Usage:
    import macos_use.ax as ax

    # Get screen size
    width, height = ax.GetScreenSize()

    # Click at coordinates
    ax.Click(100, 200)

    # Create a Control from PID
    app = ax.Control(pid=12345)
    window = app.FocusedWindow
    buttons = window.FindAll(role=ax.Role.Button)

    # Use patterns
    for btn in buttons:
        if ax.InvokePattern.IsSupported(btn.Element):
            pattern = ax.InvokePattern(btn.Element)
            pattern.Invoke()
"""

# Enums - Constants
from .enums import (
    AXError,
    AXErrorNames,
    AXValueType,
    Role,
    RoleNames,
    INTERACTIVE_ROLES,
    CONTAINER_ROLES,
    NON_INTERACTIVE_ROLES,
    SCROLLABLE_ROLES,
    Subrole,
    SubroleNames,
    WINDOW_CONTROL_SUBROLES,
    Attribute,
    Action,
    ActionNames,
    INTERACTIVE_ACTIONS,
    Notification,
    NotificationNames,
    NotificationKey,
    FOCUS_NOTIFICATIONS,
    STRUCTURE_NOTIFICATIONS,
    PROPERTY_NOTIFICATIONS,
    ALL_NOTIFICATIONS,
    KeyCode,
    KEY_NAME_TO_CODE,
    MouseEventType,
    MouseButton,
    EventFlag,
    MODIFIER_KEY_MAP,
    Orientation,
    SortDirection,
    Units,
    TextAttribute,
    ActivationPolicy,
    ActivationPolicyNames,
)

# Core - Low-level functions
from .core import (
    Rect,
    Point,
    Size,
    _AXClient,
    GetRootControl,
    ControlFromPID,
    IsAccessibilityEnabled,
    IsAccessibilityEnabledWithPrompt,
    GetAttribute,
    SetAttribute,
    IsAttributeSettable,
    GetAttributeNames,
    GetActionNames,
    PerformAction,
    GetChildCount,
    GetChildren,
    GetPosition,
    GetSize,
    GetRect,
    ElementAtPosition,
    GetElementPid,
    GetMultipleAttributeValues,
    GetTraversalBatch,
    GetAttributeValues,
    GetActionDescription,
    SetMessagingTimeout,
    GetMessagingTimeout,
    GetScreenSize,
    GetMainDisplaySize,
    GetDisplayCount,
    GetDisplayBounds,
    GetDPIScale,
    GetPerDisplayInfo,
    CaptureScreen,
    CGImageToPIL,
    GetCursorPos,
    SetCursorPos,
    MoveTo,
    Click,
    RightClick,
    MiddleClick,
    DoubleClick,
    DragTo,
    WheelDown,
    WheelUp,
    WheelLeft,
    WheelRight,
    KeyDown,
    KeyUp,
    KeyPress,
    HotKey,
    TypeText,
    GetWindowList,
    GetForegroundWindowPID,
    GetFrontmostApplication,
    GetForegroundControl,
    GetFocusedControl,
    GetRunningApplications,
    GetRunningApplicationByName,
    GetRunningApplicationByBundleId,
    ActivateApplication,
    LaunchApplication,
    HideOtherApplications,
    GetMenuBarOwningApplication,
    GetApplicationPathByName,
    GetApplicationPathByBundleID,
    OpenFile,
    OpenURL,
    SelectFileInFinder,
    RecycleFiles,
    DuplicateFiles,
    IsFilePackage,
    GetIconForFile,
    GetIconForFileType,
    GetIconForFiles,
    GetFileInfo,
    GetLocalizedDescriptionForType,
    GetDesktopImageURL,
    SetDesktopImage,
    GetWorkspaceNotificationCenter,
    GetMacOSVersion,
    GetDefaultLanguage,
    ExecuteCommand,
)

# Controls - Element wrappers
from .controls import (
    Control,
    CreateControl,
    ApplicationControl,
    WindowControl,
    ButtonControl,
    CheckBoxControl,
    RadioButtonControl,
    TextFieldControl,
    TextAreaControl,
    ComboBoxControl,
    PopUpButtonControl,
    SliderControl,
    MenuItemControl,
    MenuBarItemControl,
    TabControl,
    ListControl,
    TableControl,
    OutlineControl,
    ScrollAreaControl,
    GroupControl,
    ImageControl,
    LinkControl,
    ProgressIndicatorControl,
    StaticTextControl,
    WebAreaControl,
    DisclosureTriangleControl,
    DockItemControl,
    CellControl,
    RowControl,
)

# Patterns - Interaction patterns
from .patterns import (
    InvokePattern,
    ValuePattern,
    RangeValuePattern,
    TogglePattern,
    ExpandCollapsePattern,
    ScrollPattern,
    SelectionPattern,
    WindowPattern,
    TextPattern,
    GetPattern,
)

# Events - Observation system
from .events import (
    EventObserver,
    AppObserver,
)
