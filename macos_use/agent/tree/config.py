"""
Configuration constants for macOS accessibility tree traversal.
Defines interactive roles, non-interactive roles, and action types.
"""

# macOS Accessibility roles that represent interactive UI elements
INTERACTIVE_ROLES = {
    'AXButton',
    'AXCheckBox',
    'AXRadioButton',
    'AXTextField',
    'AXTextArea',
    'AXComboBox',
    'AXPopUpButton',
    'AXSlider',
    'AXIncrementor',
    'AXLink',
    'AXMenuItem',
    'AXMenuBarItem',
    'AXTab',
    # 'AXImage',
    'AXDockItem',
    'AXCell',
    'AXRow',
    'AXToggle',           # Toggle switches (used in Control Center)
    'AXSwitch',           # Switch elements
    'AXDisclosureTriangle',  # Expandable sections
    'AXColorWell',        # Color picker
    'AXLevelIndicator',   # Level indicators
    'AXValueIndicator',   # Value indicators (like brightness)
}


# Roles that are containers and should not be assigned as interactive elements
NON_INTERACTIVE_ROLES = {
    'AXList',
    'AXMenuBar',
    'AXMenu',
    'AXGroup',
    'AXScrollArea',
    'AXStaticText',
    'AXRadioGroup',
    'AXGrid',
    'AXApplication',
    'AXWindow',
    'AXToolbar',
    'AXSplitGroup',
    'AXTabGroup',
    'AXWebArea',
}

# Common accessibility actions
class Actions:
    PRESS = "AXPress"
    INCREMENT = "AXIncrement"
    DECREMENT = "AXDecrement"
    CONFIRM = "AXConfirm"
    CANCEL = "AXCancel"
    SHOW_MENU = "AXShowMenu"
    PICK = "AXPick"
    RAISE = "AXRaise"

# Actions that indicate an element is interactive
INTERACTIVE_ACTIONS = {
    Actions.PRESS,
    Actions.CONFIRM,
    Actions.CANCEL,
    Actions.INCREMENT,
    Actions.DECREMENT,
    Actions.SHOW_MENU,
    Actions.PICK,
    Actions.RAISE,
}

# Subroles for window controls
WINDOW_CONTROL_SUBROLES = {
    'AXCloseButton': 'Close Button',
    'AXMinimizeButton': 'Minimize Button',
    'AXZoomButton': 'Zoom Button',
    'AXFullScreenButton': 'Full Screen Button',
}

# Roles that represent scrollable containers
SCROLLABLE_ROLES = {
    'AXScrollArea',    # Standard scroll areas
    'AXScrollView',    # Scroll views (alternative name)
    'AXTable',         # Tables with scrollable content
    'AXList',          # Lists that can scroll
    'AXOutline',       # Outline views (like tree views)
    'AXBrowser',       # Column browser views
    'AXTextArea',      # Multi-line text areas can scroll
}
