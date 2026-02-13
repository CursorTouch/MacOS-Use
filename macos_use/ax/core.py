"""
Core macOS Accessibility API functions.
Provides low-level access to the AX framework, screen management,
mouse/keyboard input simulation, and window management.

Equivalent to the Windows UIA core.py module, adapted for macOS.
Uses native Quartz CGEvent APIs instead of pyautogui for input simulation.
"""

import time
import subprocess
import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import Quartz
from Quartz import (
    CGEventCreateMouseEvent,
    CGEventCreateKeyboardEvent,
    CGEventCreateScrollWheelEvent,
    CGEventPost,
    CGEventSetFlags,
    CGEventSetIntegerValueField,
    CGEventKeyboardSetUnicodeString,
    kCGHIDEventTap,
    kCGEventMouseMoved,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGEventLeftMouseDragged,
    kCGEventRightMouseDown,
    kCGEventRightMouseUp,
    kCGEventOtherMouseDown,
    kCGEventOtherMouseUp,
    kCGScrollEventUnitPixel,
    kCGScrollEventUnitLine,
    kCGMouseButtonLeft,
    kCGMouseButtonRight,
    kCGMouseButtonCenter,
    kCGEventFlagMaskCommand,
    kCGEventFlagMaskShift,
    kCGEventFlagMaskAlternate,
    kCGEventFlagMaskControl,
    CGMainDisplayID,
    CGDisplayPixelsWide,
    CGDisplayPixelsHigh,
    CGDisplayBounds,
    CGGetActiveDisplayList,
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGWindowListExcludeDesktopElements,
    kCGNullWindowID,
    CGRectInfinite,
    kCGWindowImageDefault,
    kCGWindowListOptionAll,
)
from Quartz.CoreGraphics import (
    CGImageGetWidth,
    CGImageGetHeight,
    CGImageGetBytesPerRow,
    CGDataProviderCopyData,
    CGImageGetDataProvider,
)
from ApplicationServices import (
    AXUIElementCreateSystemWide,
    AXUIElementCreateApplication,
    AXUIElementCopyAttributeValue,
    AXUIElementCopyAttributeNames,
    AXUIElementSetAttributeValue,
    AXUIElementPerformAction,
    AXUIElementGetAttributeValueCount,
    AXUIElementCopyActionNames,
    AXUIElementIsAttributeSettable,
    AXIsProcessTrusted,
    kAXErrorSuccess,
)
from Cocoa import NSWorkspace

from .enums import (
    AXError,
    Attribute,
    KeyCode,
    KEY_NAME_TO_CODE,
    MODIFIER_KEY_MAP,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Geometry Data Classes
# =============================================================================

@dataclass
class Rect:
    """
    Rectangle representing a UI element's bounds.
    Equivalent to Windows UIA RECT.
    """
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center(self) -> Tuple[float, float]:
        return (
            self.left + self.width / 2,
            self.top + self.height / 2,
        )

    @classmethod
    def from_position_size(cls, x: float, y: float, w: float, h: float) -> 'Rect':
        return cls(left=x, top=y, right=x + w, bottom=y + h)

    def intersects(self, other: 'Rect') -> bool:
        return not (
            self.right < other.left or
            self.left > other.right or
            self.bottom < other.top or
            self.top > other.bottom
        )

    def intersection(self, other: 'Rect') -> Optional['Rect']:
        new_left = max(self.left, other.left)
        new_top = max(self.top, other.top)
        new_right = min(self.right, other.right)
        new_bottom = min(self.bottom, other.bottom)
        if new_left < new_right and new_top < new_bottom:
            return Rect(left=new_left, top=new_top, right=new_right, bottom=new_bottom)
        return None

    def __str__(self) -> str:
        return f'Rect(left={int(self.left)}, top={int(self.top)}, right={int(self.right)}, bottom={int(self.bottom)})'


@dataclass
class Point:
    """A 2D point in screen coordinates."""
    x: float
    y: float

    def __str__(self) -> str:
        return f'({int(self.x)}, {int(self.y)})'


@dataclass
class Size:
    """A 2D size."""
    width: float
    height: float

    def __str__(self) -> str:
        return f'({int(self.width)}, {int(self.height)})'


# =============================================================================
# AX Client Singleton
# =============================================================================

class _AXClient:
    """
    Singleton providing access to the macOS Accessibility API.
    Equivalent to Windows UIA _AutomationClient.
    """
    _instance: Optional['_AXClient'] = None

    @classmethod
    def instance(cls) -> '_AXClient':
        """Get or create the singleton AX client instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._system_wide = AXUIElementCreateSystemWide()
        self._is_trusted = AXIsProcessTrusted()
        if not self._is_trusted:
            logger.warning(
                "Accessibility access is not granted. "
                "Enable it in System Settings > Privacy & Security > Accessibility."
            )

    @property
    def system_wide(self):
        """Get the system-wide accessibility element."""
        return self._system_wide

    @property
    def is_trusted(self) -> bool:
        """Check if the process has accessibility permissions."""
        return self._is_trusted


# =============================================================================
# Element Creation Functions
# =============================================================================

def GetRootControl():
    """
    Get the system-wide root accessibility element.
    Equivalent to Windows UIA GetRootControl().
    """
    return _AXClient.instance().system_wide


def ControlFromPID(pid: int):
    """
    Create an accessibility element for the application with the given PID.
    Equivalent to Windows UIA ControlFromHandle().
    """
    return AXUIElementCreateApplication(pid)


def IsAccessibilityEnabled() -> bool:
    """Check if accessibility access has been granted for this process."""
    return AXIsProcessTrusted()


# =============================================================================
# Attribute Access Helpers
# =============================================================================

def GetAttribute(element, attribute: str):
    """
    Get an attribute value from an AXUIElement.
    Returns None if the attribute is not available or an error occurs.
    """
    try:
        error, value = AXUIElementCopyAttributeValue(element, attribute, None)
        if error == kAXErrorSuccess:
            return value
    except Exception:
        pass
    return None


def SetAttribute(element, attribute: str, value) -> bool:
    """
    Set an attribute value on an AXUIElement.
    Returns True if successful.
    """
    try:
        error = AXUIElementSetAttributeValue(element, attribute, value)
        return error == kAXErrorSuccess
    except Exception:
        return False


def IsAttributeSettable(element, attribute: str) -> bool:
    """Check if an attribute can be set on an element."""
    try:
        error, settable = AXUIElementIsAttributeSettable(element, attribute, None)
        if error == kAXErrorSuccess:
            return bool(settable)
    except Exception:
        pass
    return False


def GetAttributeNames(element) -> list:
    """Get all attribute names supported by an element."""
    try:
        error, names = AXUIElementCopyAttributeNames(element, None)
        if error == kAXErrorSuccess and names:
            return list(names)
    except Exception:
        pass
    return []


def GetActionNames(element) -> list:
    """Get all action names supported by an element."""
    try:
        error, names = AXUIElementCopyActionNames(element, None)
        if error == kAXErrorSuccess and names:
            return list(names)
    except Exception:
        pass
    return []


def PerformAction(element, action: str) -> bool:
    """
    Perform an action on an AXUIElement.
    Returns True if successful.
    """
    try:
        error = AXUIElementPerformAction(element, action)
        return error == kAXErrorSuccess
    except Exception:
        return False


def GetChildCount(element) -> int:
    """Get the number of children of an element."""
    try:
        error, count = AXUIElementGetAttributeValueCount(element, Attribute.Children, None)
        if error == kAXErrorSuccess:
            return count
    except Exception:
        pass
    return 0


def GetChildren(element) -> list:
    """Get child elements of an accessibility element."""
    try:
        error, count = AXUIElementGetAttributeValueCount(element, Attribute.Children, None)
        if error != kAXErrorSuccess or count == 0:
            return []
        error, children = AXUIElementCopyAttributeValue(element, Attribute.Children, None)
        if error == kAXErrorSuccess and children:
            return list(children)
    except Exception:
        pass
    return []


def GetPosition(element) -> Optional[Tuple[float, float]]:
    """Get the position (x, y) of an accessibility element in screen coordinates."""
    error, pos_val = AXUIElementCopyAttributeValue(element, Attribute.Position, None)
    if error != kAXErrorSuccess or pos_val is None:
        return None

    # Try standard attribute access (bridged CGPoint/NSPoint)
    if hasattr(pos_val, 'x') and hasattr(pos_val, 'y'):
        return (pos_val.x, pos_val.y)

    # Try AXValue string parsing
    if hasattr(pos_val, 'getValue_size_type_') or str(pos_val).startswith('<AXValue'):
        desc = str(pos_val)
        try:
            match = re.search(r'x[:=]\s*([-\d\.]+).*?y[:=]\s*([-\d\.]+)', desc, re.IGNORECASE)
            if match:
                return (float(match.group(1)), float(match.group(2)))
        except Exception:
            pass

    # Try generic sequence access
    try:
        if len(pos_val) == 2:
            return (pos_val[0], pos_val[1])
    except Exception:
        pass

    return None


def GetSize(element) -> Optional[Tuple[float, float]]:
    """Get the size (width, height) of an accessibility element."""
    error, size_val = AXUIElementCopyAttributeValue(element, Attribute.Size, None)
    if error != kAXErrorSuccess or size_val is None:
        return None

    # Try standard attribute access (bridged CGSize/NSSize)
    if hasattr(size_val, 'width') and hasattr(size_val, 'height'):
        return (size_val.width, size_val.height)

    # Try generic sequence access
    try:
        if len(size_val) == 2:
            return (size_val[0], size_val[1])
    except Exception:
        pass

    # Try AXValue string parsing
    if hasattr(size_val, 'getValue_size_type_') or str(size_val).startswith('<AXValue'):
        desc = str(size_val)
        try:
            match = re.search(r'w(idth)?[:=]\s*([-\d\.]+).*?h(eight)?[:=]\s*([-\d\.]+)', desc, re.IGNORECASE)
            if match:
                return (float(match.group(2)), float(match.group(4)))
        except Exception:
            pass

    return None


def GetRect(element) -> Optional[Rect]:
    """Get the bounding rectangle of an accessibility element."""
    pos = GetPosition(element)
    size = GetSize(element)
    if pos and size:
        return Rect.from_position_size(pos[0], pos[1], size[0], size[1])
    return None


# =============================================================================
# Screen Functions
# =============================================================================

def GetScreenSize() -> Tuple[int, int]:
    """
    Get the combined resolution of all active displays (virtual screen size).
    Returns (width, height).
    Equivalent to Windows GetSystemMetrics.
    """
    try:
        max_displays = 32
        res = CGGetActiveDisplayList(max_displays, None, None)
        if res and res[1]:
            display_ids = res[1]
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
            for display_id in display_ids:
                bounds = CGDisplayBounds(display_id)
                x = bounds.origin.x
                y = bounds.origin.y
                w = bounds.size.width
                h = bounds.size.height
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
            return (int(max_x - min_x), int(max_y - min_y))
    except Exception as e:
        logger.warning(f"Failed to calculate virtual screen size: {e}")

    # Fallback to main display
    main_display = CGMainDisplayID()
    width = CGDisplayPixelsWide(main_display)
    height = CGDisplayPixelsHigh(main_display)
    return (width, height)


def GetMainDisplaySize() -> Tuple[int, int]:
    """Get the resolution of the main display. Returns (width, height)."""
    main_display = CGMainDisplayID()
    return (CGDisplayPixelsWide(main_display), CGDisplayPixelsHigh(main_display))


def GetDisplayCount() -> int:
    """Get the number of active displays."""
    try:
        res = CGGetActiveDisplayList(32, None, None)
        if res and res[1]:
            return len(res[1])
    except Exception:
        pass
    return 1


def GetDisplayBounds() -> list[Rect]:
    """Get the bounding rectangles of all active displays."""
    rects = []
    try:
        res = CGGetActiveDisplayList(32, None, None)
        if res and res[1]:
            for display_id in res[1]:
                bounds = CGDisplayBounds(display_id)
                rects.append(Rect(
                    left=bounds.origin.x,
                    top=bounds.origin.y,
                    right=bounds.origin.x + bounds.size.width,
                    bottom=bounds.origin.y + bounds.size.height,
                ))
    except Exception:
        pass
    return rects


def GetDPIScale() -> float:
    """
    Get the DPI scale factor of the main display.
    Returns 2.0 for Retina displays, 1.0 for standard.
    """
    try:
        main_display = CGMainDisplayID()
        pixel_width = CGDisplayPixelsWide(main_display)
        bounds = CGDisplayBounds(main_display)
        point_width = bounds.size.width
        if point_width > 0:
            return round(pixel_width / point_width, 1)
    except Exception:
        pass
    return 1.0


# =============================================================================
# Screenshot Functions
# =============================================================================

def CaptureScreen(rect=None):
    """
    Capture a screenshot of the screen.
    Returns a CGImage, or None on failure.

    Args:
        rect: Optional Quartz CGRect to capture. If None, captures entire screen.
    """
    try:
        capture_rect = rect if rect is not None else CGRectInfinite
        cg_image = Quartz.CGWindowListCreateImage(
            capture_rect,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
            kCGWindowImageDefault,
        )
        return cg_image
    except Exception as e:
        logger.error(f"Screenshot capture failed: {e}")
        return None


def CGImageToPIL(cg_image):
    """
    Convert a CGImage to a PIL Image.
    Requires Pillow to be installed.
    """
    from PIL import Image
    width = CGImageGetWidth(cg_image)
    height = CGImageGetHeight(cg_image)
    bytes_per_row = CGImageGetBytesPerRow(cg_image)
    pixel_data = CGDataProviderCopyData(CGImageGetDataProvider(cg_image))
    return Image.frombuffer(
        "RGBA", (width, height), pixel_data, "raw", "BGRA", bytes_per_row, 1
    )


# =============================================================================
# Mouse Functions
# =============================================================================

def GetCursorPos() -> Tuple[int, int]:
    """
    Get the current mouse cursor position.
    Returns (x, y) in screen coordinates.
    """
    event = Quartz.CGEventCreate(None)
    point = Quartz.CGEventGetLocation(event)
    return (int(point.x), int(point.y))


def SetCursorPos(x: int, y: int) -> None:
    """
    Move the mouse cursor to the specified position.
    Equivalent to Windows SetCursorPos.
    """
    event = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, event)


def MoveTo(x: int, y: int) -> None:
    """Move the mouse cursor to the specified coordinates."""
    SetCursorPos(x, y)


def Click(x: int, y: int, waitTime: float = 0.05) -> None:
    """
    Perform a left mouse click at the specified coordinates.
    Equivalent to Windows UIA Click().
    """
    event_down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
    event_up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, event_down)
    time.sleep(waitTime)
    CGEventPost(kCGHIDEventTap, event_up)


def RightClick(x: int, y: int, waitTime: float = 0.05) -> None:
    """Perform a right mouse click at the specified coordinates."""
    event_down = CGEventCreateMouseEvent(None, kCGEventRightMouseDown, (x, y), kCGMouseButtonRight)
    event_up = CGEventCreateMouseEvent(None, kCGEventRightMouseUp, (x, y), kCGMouseButtonRight)
    CGEventPost(kCGHIDEventTap, event_down)
    time.sleep(waitTime)
    CGEventPost(kCGHIDEventTap, event_up)


def MiddleClick(x: int, y: int, waitTime: float = 0.05) -> None:
    """Perform a middle mouse click at the specified coordinates."""
    event_down = CGEventCreateMouseEvent(None, kCGEventOtherMouseDown, (x, y), kCGMouseButtonCenter)
    event_up = CGEventCreateMouseEvent(None, kCGEventOtherMouseUp, (x, y), kCGMouseButtonCenter)
    CGEventPost(kCGHIDEventTap, event_down)
    time.sleep(waitTime)
    CGEventPost(kCGHIDEventTap, event_up)


def DoubleClick(x: int, y: int, waitTime: float = 0.05) -> None:
    """Perform a double left-click at the specified coordinates."""
    event_down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
    CGEventSetIntegerValueField(event_down, Quartz.kCGMouseEventClickState, 2)
    event_up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
    CGEventSetIntegerValueField(event_up, Quartz.kCGMouseEventClickState, 2)
    CGEventPost(kCGHIDEventTap, event_down)
    time.sleep(waitTime)
    CGEventPost(kCGHIDEventTap, event_up)


def DragTo(start_x: int, start_y: int, end_x: int, end_y: int,
           duration: float = 0.5, steps: int = 20) -> None:
    """
    Perform a mouse drag from start to end position.
    """
    # Mouse down at start
    event_down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (start_x, start_y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, event_down)
    time.sleep(0.05)

    # Smooth drag
    step_delay = duration / steps
    for i in range(1, steps + 1):
        progress = i / steps
        cx = start_x + (end_x - start_x) * progress
        cy = start_y + (end_y - start_y) * progress
        event_drag = CGEventCreateMouseEvent(None, kCGEventLeftMouseDragged, (cx, cy), kCGMouseButtonLeft)
        CGEventPost(kCGHIDEventTap, event_drag)
        time.sleep(step_delay)

    # Mouse up at end
    event_up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (end_x, end_y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, event_up)


def WheelDown(clicks: int = 1, interval: float = 0.05) -> None:
    """
    Scroll down by the specified number of clicks.
    Equivalent to Windows mouse_event with WHEEL.
    """
    for _ in range(clicks):
        event = CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitLine, 1, -3)
        CGEventPost(kCGHIDEventTap, event)
        time.sleep(interval)


def WheelUp(clicks: int = 1, interval: float = 0.05) -> None:
    """Scroll up by the specified number of clicks."""
    for _ in range(clicks):
        event = CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitLine, 1, 3)
        CGEventPost(kCGHIDEventTap, event)
        time.sleep(interval)


def WheelLeft(clicks: int = 1, interval: float = 0.05) -> None:
    """Scroll left by the specified number of clicks."""
    for _ in range(clicks):
        event = CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitLine, 2, 0, 3)
        CGEventPost(kCGHIDEventTap, event)
        time.sleep(interval)


def WheelRight(clicks: int = 1, interval: float = 0.05) -> None:
    """Scroll right by the specified number of clicks."""
    for _ in range(clicks):
        event = CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitLine, 2, 0, -3)
        CGEventPost(kCGHIDEventTap, event)
        time.sleep(interval)


# =============================================================================
# Keyboard Functions
# =============================================================================

def KeyDown(key_code: int, flags: int = 0) -> None:
    """
    Press a key down.

    Args:
        key_code: Virtual key code from KeyCode class.
        flags: Modifier flags from EventFlag class (0 = no modifiers).
    """
    event = CGEventCreateKeyboardEvent(None, key_code, True)
    # Always set flags explicitly — CGEventCreateKeyboardEvent(None, ...)
    # inherits the current system modifier state, so we must override it
    # even when flags=0 to prevent stale modifiers from leaking through.
    CGEventSetFlags(event, flags)
    CGEventPost(kCGHIDEventTap, event)


def KeyUp(key_code: int, flags: int = 0) -> None:
    """Release a key."""
    event = CGEventCreateKeyboardEvent(None, key_code, False)
    # Always clear/set flags explicitly to prevent modifier leakage.
    CGEventSetFlags(event, flags)
    CGEventPost(kCGHIDEventTap, event)


def KeyPress(key_code: int, flags: int = 0, waitTime: float = 0.05) -> None:
    """Press and release a key."""
    KeyDown(key_code, flags)
    time.sleep(waitTime)
    KeyUp(key_code, flags)


def HotKey(*keys: str, waitTime: float = 0.05) -> None:
    """
    Press a keyboard shortcut using key names.
    Example: HotKey('command', 'c') for Cmd+C.

    Args:
        keys: Key names (e.g., 'command', 'shift', 'a').
        waitTime: Delay between key down and key up.
    """
    # Build modifier flags and find the main key
    flags = 0
    main_key_code = None

    for key in keys:
        key_lower = key.lower().strip()
        if key_lower in MODIFIER_KEY_MAP:
            flags |= MODIFIER_KEY_MAP[key_lower]
        elif key_lower in KEY_NAME_TO_CODE:
            main_key_code = KEY_NAME_TO_CODE[key_lower]
        else:
            logger.warning(f"Unknown key: {key}")

    if main_key_code is not None:
        KeyPress(main_key_code, flags, waitTime)
    elif flags:
        # Only modifiers pressed (e.g., just pressing Command)
        # Press and release each modifier
        for key in keys:
            key_lower = key.lower().strip()
            if key_lower in KEY_NAME_TO_CODE:
                KeyDown(KEY_NAME_TO_CODE[key_lower])
        time.sleep(waitTime)
        for key in reversed(keys):
            key_lower = key.lower().strip()
            if key_lower in KEY_NAME_TO_CODE:
                KeyUp(KEY_NAME_TO_CODE[key_lower])


def _release_modifiers() -> None:
    """
    Release all modifier keys to ensure a clean keyboard state.

    This prevents stale modifier flags (e.g. from a preceding HotKey call)
    from leaking into subsequent key events. Sends explicit key-up events
    for Command, Shift, Option, and Control on both left and right sides.
    """
    modifier_keycodes = [
        KeyCode.Command, KeyCode.Shift, KeyCode.Option, KeyCode.Control,
        KeyCode.RightCommand, KeyCode.RightShift, KeyCode.RightOption, KeyCode.RightControl,
    ]
    for kc in modifier_keycodes:
        event = CGEventCreateKeyboardEvent(None, kc, False)
        CGEventSetFlags(event, 0)
        CGEventPost(kCGHIDEventTap, event)


def TypeText(text: str, interval: float = 0.01) -> None:
    """
    Type a string of text using native CGEvent keyboard events.
    Uses CGEventKeyboardSetUnicodeString for natural text input that
    supports all Unicode scripts (Hindi, Chinese, Arabic, etc.)
    without touching the system clipboard.

    Each character is typed as an individual key-down/key-up pair,
    simulating real keystrokes. For ASCII characters with known key
    codes, real virtual key events are generated. For all other
    characters (Unicode), CGEventKeyboardSetUnicodeString is used to
    inject the character directly into the keyboard event stream.

    Args:
        text: The text to type.
        interval: Delay between keystrokes in seconds.
    """
    if not text:
        return

    # Release all modifier keys before typing to ensure no stale
    # Command/Shift/Option/Control state leaks into the key events.
    # This is critical after HotKey calls (e.g. Cmd+A for select-all)
    # which may leave the system thinking a modifier is still held.
    _release_modifiers()
    time.sleep(0.02)

    for char in text:
        _type_character(char, interval)


def _type_character(char: str, interval: float = 0.01) -> None:
    """
    Type a single character using CGEvent keyboard simulation.

    For ASCII characters with known virtual key codes, generates real
    key press events (most natural for applications). For everything
    else, uses CGEventKeyboardSetUnicodeString to inject the character
    natively without clipboard involvement.
    """
    key_lower = char.lower()
    if key_lower in KEY_NAME_TO_CODE:
        key_code = KEY_NAME_TO_CODE[key_lower]
        flags = 0
        # Apply shift for uppercase letters
        if char.isupper():
            flags = kCGEventFlagMaskShift
        KeyPress(key_code, flags, interval)
    elif char == ' ':
        KeyPress(KeyCode.Space, 0, interval)
    elif char == '\n':
        KeyPress(KeyCode.Return, 0, interval)
    elif char == '\t':
        KeyPress(KeyCode.Tab, 0, interval)
    else:
        # Unicode character — use CGEventKeyboardSetUnicodeString
        _type_unicode_char(char)
        time.sleep(interval)


def _type_unicode_char(char: str) -> None:
    """
    Type a Unicode character using CGEventKeyboardSetUnicodeString.

    This injects the character directly into the keyboard event stream
    without touching the clipboard. Works with any Unicode script:
    Devanagari (Hindi), CJK (Chinese/Japanese/Korean), Arabic, Cyrillic,
    emoji, and all other Unicode characters.

    The character is encoded as UTF-16 (macOS native UniChar format) and
    attached to a CGEvent keyboard event pair (key-down + key-up).
    """
    # Encode to UTF-16LE to get the UniChar representation
    # Each UniChar is 2 bytes; surrogate pairs (e.g. emoji) produce 2 UniChars
    utf16_bytes = char.encode('utf-16-le')
    utf16_length = len(utf16_bytes) // 2  # Number of UniChar code units

    # Key down event with the Unicode string attached
    # Use key code 0 as a placeholder — the actual character comes from
    # CGEventKeyboardSetUnicodeString, not from the virtual key code.
    event_down = CGEventCreateKeyboardEvent(None, 0, True)
    CGEventSetFlags(event_down, 0)  # Explicitly clear all modifier flags
    CGEventKeyboardSetUnicodeString(event_down, utf16_length, char)
    CGEventPost(kCGHIDEventTap, event_down)

    # Key up event (completes the keystroke pair)
    event_up = CGEventCreateKeyboardEvent(None, 0, False)
    CGEventSetFlags(event_up, 0)  # Explicitly clear all modifier flags
    CGEventPost(kCGHIDEventTap, event_up)


# =============================================================================
# Window Functions
# =============================================================================

def GetWindowList(on_screen_only: bool = True) -> list:
    """
    Get list of window info dictionaries from the window server.
    Returns raw CGWindowListCopyWindowInfo results.
    """
    options = kCGWindowListOptionOnScreenOnly if on_screen_only else kCGWindowListOptionAll
    if on_screen_only:
        options |= kCGWindowListExcludeDesktopElements
    window_list = CGWindowListCopyWindowInfo(options, kCGNullWindowID)
    return list(window_list) if window_list else []


def GetForegroundWindowPID() -> Optional[int]:
    """
    Get the PID of the frontmost application using CGWindowListCopyWindowInfo.
    More reliable than NSWorkspace when no NSRunLoop is active.
    Equivalent to Windows GetForegroundWindow().
    """
    window_list = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
        kCGNullWindowID
    )
    if not window_list:
        return None

    for win_info in window_list:
        # Normal windows live at layer 0
        layer = win_info.get(Quartz.kCGWindowLayer, -1)
        if layer != 0:
            continue
        pid = win_info.get(Quartz.kCGWindowOwnerPID, 0)
        if pid:
            return pid
    return None


def GetRunningApplications() -> list:
    """Get all running applications from NSWorkspace."""
    return list(NSWorkspace.sharedWorkspace().runningApplications())


def GetFrontmostApplication():
    """Get the frontmost application from NSWorkspace."""
    return NSWorkspace.sharedWorkspace().frontmostApplication()


def ActivateApplication(pid: int) -> bool:
    """
    Activate (bring to front) an application by PID.
    """
    from Cocoa import NSApplicationActivateIgnoringOtherApps
    workspace = NSWorkspace.sharedWorkspace()
    for app in workspace.runningApplications():
        if app.processIdentifier() == pid:
            return app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
    return False


def LaunchApplication(name: str) -> bool:
    """
    Launch an application by name.
    Returns True if successful.
    """
    try:
        subprocess.run(['open', '-a', name], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        pass

    # Fallback to NSWorkspace
    workspace = NSWorkspace.sharedWorkspace()
    return workspace.launchApplication_(name)


# =============================================================================
# System Info Functions
# =============================================================================

def GetMacOSVersion() -> str:
    """Get the macOS version string (e.g., 'macOS 15.3')."""
    try:
        result = subprocess.run(['sw_vers', '-productVersion'], capture_output=True, text=True)
        version = result.stdout.strip()
        name_result = subprocess.run(['sw_vers', '-productName'], capture_output=True, text=True)
        name = name_result.stdout.strip()
        return f"{name} {version}"
    except Exception:
        return "macOS"


def GetDefaultLanguage() -> str:
    """Get the default system language."""
    try:
        result = subprocess.run(
            ['defaults', 'read', '-g', 'AppleLanguages'],
            capture_output=True, text=True
        )
        langs = result.stdout.strip()
        if langs.startswith('('):
            first_lang = langs.split(',')[0].strip('() "')
            return first_lang
        return "en-US"
    except Exception:
        return "en-US"


def ExecuteCommand(command: str, mode: str = 'shell', timeout: int = 10) -> Tuple[str, int]:
    """
    Execute a command in shell or osascript mode.

    Args:
        command: Command to execute.
        mode: 'shell' for bash, 'osascript' for AppleScript.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (output, return_code).
    """
    import os
    env = os.environ.copy()
    try:
        if mode == 'osascript':
            result = subprocess.run(
                ['osascript', '-e', command],
                capture_output=True, text=True, timeout=timeout, env=env
            )
        else:
            result = subprocess.run(
                command, shell=True,
                capture_output=True, text=True, timeout=timeout, env=env
            )
        output = result.stdout or result.stderr or ''
        return (output.strip(), result.returncode)
    except subprocess.TimeoutExpired:
        return (f"Command timed out after {timeout} seconds", -1)
    except Exception as e:
        return (str(e), -1)
