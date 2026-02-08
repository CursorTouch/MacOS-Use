"""
Tree service for macOS accessibility tree traversal.
Provides methods to capture and traverse the accessibility tree using the macOS Accessibility API.
Uses multi-threading for parallel traversal of different UI regions.
"""
from ApplicationServices import (
    AXUIElementCopyAttributeValue,
    AXUIElementGetAttributeValueCount,
    AXUIElementCreateApplication,
    kAXErrorSuccess,
    kAXChildrenAttribute,
    kAXRoleAttribute,
    kAXSubroleAttribute,
    kAXTitleAttribute,
    kAXDescriptionAttribute,
    kAXPositionAttribute,
    kAXSizeAttribute,
    kAXFocusedWindowAttribute,
    kAXMainWindowAttribute,
    kAXWindowsAttribute,
    kAXMenuBarAttribute,
)
from Cocoa import NSWorkspace
from macos_use.agent.tree.views import (
    TreeState,
    TreeElementNode,
    ScrollElementNode,
    BoundingBox,
    Center,
)
from macos_use.agent.tree.config import (
    INTERACTIVE_ROLES,
    NON_INTERACTIVE_ROLES,
    INTERACTIVE_ACTIONS,
    WINDOW_CONTROL_SUBROLES,
    SCROLLABLE_ROLES,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, TYPE_CHECKING
from threading import Lock
import weakref
import logging
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if TYPE_CHECKING:
    from macos_use.agent.desktop.service import Desktop


class Tree:
    """
    Tree class for traversing the macOS accessibility tree.
    Captures interactive and scrollable elements for desktop automation.
    Uses multi-threading for parallel traversal of different UI regions.
    """

    def __init__(self, desktop: 'Desktop'):
        self.desktop = weakref.proxy(desktop)
        self._lock = Lock()

    def get_attr(self, element, attr: str):
        """Get an attribute value from an accessibility element."""
        try:
            error, value = AXUIElementCopyAttributeValue(element, attr, None)
            if error == kAXErrorSuccess:
                return value
        except Exception:
            pass
        return None

    def get_children(self, element) -> list:
        """Get child elements of an accessibility element."""
        try:
            error, count = AXUIElementGetAttributeValueCount(element, kAXChildrenAttribute, None)
            if error != kAXErrorSuccess or count == 0:
                return []
            
            error, children = AXUIElementCopyAttributeValue(element, kAXChildrenAttribute, None)
            if error == kAXErrorSuccess and children:
                return list(children)
        except Exception:
            pass
        return []

    def get_position(self, element) -> Optional[tuple[float, float]]:
        """Get the position (x, y) of an accessibility element."""
        error, pos_val = AXUIElementCopyAttributeValue(element, kAXPositionAttribute, None)
        if error != 0 or pos_val is None:
            return None
            
        # Try standard attribute access (for bridged CGPoint/NSPoint)
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

    def get_size(self, element) -> Optional[tuple[float, float]]:
        """Get the size (width, height) of an accessibility element."""
        error, size_val = AXUIElementCopyAttributeValue(element, kAXSizeAttribute, None)
        if error != 0 or size_val is None:
            return None
            
        # Try standard attribute access (for bridged CGSize/NSSize)
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

    def has_interactive_actions(self, actions) -> bool:
        """Check if element has interactive actions."""
        if not actions:
            return False
        return any(action in INTERACTIVE_ACTIONS for action in actions)

    def is_launchpad_open(self) -> bool:
        """
        Check if Launchpad is currently active (visible).
        Launchpad is a special overlay that hides the desktop.
        """
        apps = NSWorkspace.sharedWorkspace().runningApplications()
        dock_app = next((app for app in apps if app.localizedName() == "Dock"), None)
        if not dock_app:
            return False
        
        pid = dock_app.processIdentifier()
        ax_dock = AXUIElementCreateApplication(pid)
        children = self.get_children(ax_dock)
        
        for child in children:
            role = self.get_attr(child, kAXRoleAttribute)
            title = self.get_attr(child, kAXTitleAttribute)
            if role == 'AXGroup' and title == 'Launchpad':
                # Found the Launchpad group. Check if it's visible.
                is_hidden = self.get_attr(child, 'AXHidden')
                if is_hidden is not True:
                    return True
        return False

    def is_window_fullscreen(self, window) -> bool:
        """
        Check if a window is in fullscreen mode.
        Uses AXFullScreen attribute to detect fullscreen state.
        """
        if not window:
            return False
        
        try:
            # Check AXFullScreen attribute
            fullscreen = self.get_attr(window, 'AXFullScreen')
            if fullscreen is True:
                return True
            
            # Alternative: Check subrole for fullscreen
            subrole = self.get_attr(window, kAXSubroleAttribute)
            if subrole == 'AXFullScreenWindow':
                return True
        except Exception:
            pass
        
        return False

    def get_state(self, window_name: str = '') -> TreeState:
        """
        Capture the current accessibility tree state using parallel traversal.
        Returns a TreeState with interactive and scrollable elements.
        """
        interactive_elements: list[TreeElementNode] = []
        scrollable_elements: list[ScrollElementNode] = []

        # Get all running apps
        apps = NSWorkspace.sharedWorkspace().runningApplications()

        # Check if Launchpad is open - if so, only scan Launchpad and Dock
        launchpad_active = self.is_launchpad_open()

        # Prepare tasks for parallel execution
        tasks = []

        # Get Dock app (needed for both Launchpad detection and Dock scanning)
        dock_app = next((app for app in apps if app.bundleIdentifier() == "com.apple.dock"), None)
        
        if launchpad_active:
            # Launchpad is open - only scan Launchpad content and Dock
            logger.info("Launchpad is active - scanning only Launchpad and Dock")
            
            if dock_app:
                dock_pid = dock_app.processIdentifier()
                ax_dock = AXUIElementCreateApplication(dock_pid)
                
                # Find and scan the Launchpad group
                children = self.get_children(ax_dock)
                for child in children:
                    role = self.get_attr(child, kAXRoleAttribute)
                    title = self.get_attr(child, kAXTitleAttribute)
                    if role == 'AXGroup' and title == 'Launchpad':
                        tasks.append(('launchpad', child, 'Launchpad', None))
                        break
                
                # Also scan the Dock itself
                tasks.append(('dock', ax_dock, 'Dock', None))
        else:
            # Normal mode - scan all sources
            
            # Get frontmost application
            frontmost = NSWorkspace.sharedWorkspace().frontmostApplication()
            if not frontmost:
                logger.warning("No frontmost application found")
                return TreeState()

            pid = frontmost.processIdentifier()
            app_name = frontmost.localizedName()
            ax_app = AXUIElementCreateApplication(pid)

            # Try to get focused/main window
            focused_window = None
            error, focused_window = AXUIElementCopyAttributeValue(ax_app, kAXFocusedWindowAttribute, None)
            
            if error != kAXErrorSuccess or not focused_window:
                error, focused_window = AXUIElementCopyAttributeValue(ax_app, kAXMainWindowAttribute, None)
                
            if error != kAXErrorSuccess or not focused_window:
                error, windows = AXUIElementCopyAttributeValue(ax_app, kAXWindowsAttribute, None)
                if error == kAXErrorSuccess and windows and len(windows) > 0:
                    focused_window = windows[0]

            window_title = ''
            if focused_window:
                window_title = self.get_attr(focused_window, kAXTitleAttribute) or app_name

            # Task 1: Scan focused window
            if focused_window:
                bounds = None
                pos = self.get_position(focused_window)
                size = self.get_size(focused_window)
                if pos and size:
                    bounds = BoundingBox.from_position_size(pos[0], pos[1], size[0], size[1])
                tasks.append(('window', focused_window, window_title, bounds))

            # Task 2: Scan Dock (skip if window is fullscreen/maximized)
            is_fullscreen = self.is_window_fullscreen(focused_window)
            if dock_app and not is_fullscreen:
                dock_pid = dock_app.processIdentifier()
                ax_dock = AXUIElementCreateApplication(dock_pid)
                tasks.append(('dock', ax_dock, 'Dock', None))

            # Task 3: Scan Menu Bar of frontmost app
            error, menu_bar = AXUIElementCopyAttributeValue(ax_app, 'AXMenuBar', None)
            if error == kAXErrorSuccess and menu_bar:
                tasks.append(('menubar', menu_bar, 'MenuBar', None))

            # Task 4: Scan Control Center (WiFi, Bluetooth, etc.)
            cc_app = next((app for app in apps if app.bundleIdentifier() == "com.apple.controlcenter"), None)
            if cc_app:
                ax_cc = AXUIElementCreateApplication(cc_app.processIdentifier())
                
                # First, check for open Control Center windows/popovers
                # When Control Center is clicked, it opens a floating window with toggles
                error, cc_windows = AXUIElementCopyAttributeValue(ax_cc, kAXWindowsAttribute, None)
                if error == kAXErrorSuccess and cc_windows and len(cc_windows) > 0:
                    # Control Center has open windows - scan them for interactive elements
                    for cc_window in cc_windows:
                        tasks.append(('control_center_window', cc_window, 'Control Center', None))
                
                # Also scan the Extras Menu Bar (status items in the menu bar)
                error, cc_extras = AXUIElementCopyAttributeValue(ax_cc, "AXExtrasMenuBar", None)
                if error == kAXErrorSuccess and cc_extras:
                    tasks.append(('control_center', cc_extras, 'Control Center', None))
                
                # Scan children directly for popovers and groups
                cc_children = self.get_children(ax_cc)
                for child in cc_children:
                    child_role = self.get_attr(child, kAXRoleAttribute)
                    # Look for popovers, groups, or any visible containers
                    if child_role in ('AXPopover', 'AXGroup', 'AXSheet'):
                        is_hidden = self.get_attr(child, 'AXHidden')
                        if is_hidden is not True:
                            tasks.append(('control_center_popover', child, 'Control Center', None))

            # Task 4.5: Scan Notification Center (Date/Time, Widgets)
            nc_app = next((app for app in apps if app.bundleIdentifier() == "com.apple.notificationcenterui"), None)
            if nc_app:
                ax_nc = AXUIElementCreateApplication(nc_app.processIdentifier())
                
                # Scan visible windows (The side panel is usually a window)
                error, nc_windows = AXUIElementCopyAttributeValue(ax_nc, kAXWindowsAttribute, None)
                if error == kAXErrorSuccess and nc_windows:
                    for nc_window in nc_windows:
                        # Only scan if visible
                        is_hidden = self.get_attr(nc_window, 'AXHidden')
                        if is_hidden is not True:
                            tasks.append(('notification_center', nc_window, 'Notification Center', None))
                
                # Scan direct children (sometimes the panel is a root group)
                nc_children = self.get_children(ax_nc)
                for child in nc_children:
                    child_role = self.get_attr(child, kAXRoleAttribute)
                    if child_role in ('AXScrollArea', 'AXGroup', 'AXList'):
                        is_hidden = self.get_attr(child, 'AXHidden')
                        if is_hidden is not True:
                            tasks.append(('notification_center_child', child, 'Notification Center', None))

            # Task 5: Scan SystemUIServer (battery, volume, etc.)
            ss_app = next((app for app in apps if app.bundleIdentifier() == "com.apple.systemuiserver"), None)
            if ss_app:
                ax_ss = AXUIElementCreateApplication(ss_app.processIdentifier())
                error, ss_menu = AXUIElementCopyAttributeValue(ax_ss, kAXMenuBarAttribute, None)
                if error == kAXErrorSuccess and ss_menu:
                    tasks.append(('system_ui', ss_menu, 'SystemUI', None))

            # Task 6: Scan Spotlight
            sl_app = next((app for app in apps if app.bundleIdentifier() == "com.apple.Spotlight"), None)
            if sl_app:
                ax_sl = AXUIElementCreateApplication(sl_app.processIdentifier())
                error, sl_extras = AXUIElementCopyAttributeValue(ax_sl, "AXExtrasMenuBar", None)
                if error == kAXErrorSuccess and sl_extras:
                    tasks.append(('spotlight', sl_extras, 'Spotlight', None))

            # Task 7: Scan Desktop Icons (Finder) - only if no foreground window
            # Desktop items are only relevant when no app window is covering them
            if not focused_window:
                finder_app = next((app for app in apps if app.bundleIdentifier() == "com.apple.finder"), None)
                if finder_app:
                    ax_finder = AXUIElementCreateApplication(finder_app.processIdentifier())
                    error, finder_children = AXUIElementCopyAttributeValue(ax_finder, kAXChildrenAttribute, None)
                    if error == kAXErrorSuccess and finder_children:
                        for child in finder_children:
                            role = self.get_attr(child, kAXRoleAttribute)
                            desc = self.get_attr(child, kAXDescriptionAttribute)
                            if role == 'AXScrollArea' and desc == 'desktop':
                                tasks.append(('desktop', child, 'Desktop', None))
                                break


        # Execute traversal in parallel
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.get_appwise_nodes, element, name, bounds)
                for _, element, name, bounds in tasks
            ]

            # Collect results from all threads
            for future in as_completed(futures):
                try:
                    elements, scrollables = future.result()
                    with self._lock:
                        interactive_elements.extend(elements)
                        scrollable_elements.extend(scrollables)
                except Exception as e:
                    logger.error(f"Thread error: {e}")

        return TreeState(
            interactive_nodes=interactive_elements,
            scrollable_nodes=scrollable_elements,
        )

    def get_appwise_nodes(
        self, 
        element, 
        window_name: str,
        root_bounds: Optional[BoundingBox] = None
    ) -> tuple[list[TreeElementNode], list[ScrollElementNode]]:
        """
        Thread-safe scan for interactive elements.
        Returns collected elements instead of modifying shared state.
        """
        interactive_elements: list[TreeElementNode] = []
        scrollable_elements: list[ScrollElementNode] = []
        
        self.tree_traversal(
            element, window_name, set(),
            interactive_elements, scrollable_elements,
            root_bounds
        )
        
        return interactive_elements, scrollable_elements

    def tree_traversal(
        self,
        element,
        window_name: str,
        visited: set,
        interactive_elements: list[TreeElementNode],
        scrollable_elements: list[ScrollElementNode],
        root_bounds: Optional[BoundingBox] = None
    ):
        """
        Recursive scan for interactive elements.
        Collects elements into provided lists.
        """

        try:
            if element in visited:
                return
            visited.add(element)
        except Exception:
            pass

        # Check if hidden
        is_hidden = self.get_attr(element, 'AXHidden')
        if is_hidden is True:
            return

        role = self.get_attr(element, kAXRoleAttribute)
        title = self.get_attr(element, kAXTitleAttribute)
        description = self.get_attr(element, kAXDescriptionAttribute)
        value = self.get_attr(element, 'AXValue')
        subrole = self.get_attr(element, kAXSubroleAttribute)
        is_enabled = self.get_attr(element, 'AXEnabled')
        actions = self.get_attr(element, 'AXActions')

        pos = self.get_position(element)
        size = self.get_size(element)

        x, y, w, h = 0, 0, 0, 0
        if pos and size:
            x, y = pos
            w, h = size

        # Check if element has valid geometry
        if w <= 0 or h <= 0:
            # Skip elements with no visible size, but traverse children
            for child in self.get_children(element):
                self.tree_traversal(
                    child, window_name, visited,
                    interactive_elements, scrollable_elements,
                    root_bounds
                )
            return

        bbox = BoundingBox.from_position_size(x, y, w, h)
        if root_bounds:
            # Clip bounding box to window bounds
            clipped_bbox = bbox.intersection(root_bounds)
            if not clipped_bbox:
                # Element is completely outside window
                return
            bbox = clipped_bbox

        has_actions = self.has_interactive_actions(actions)
        
        # Get focused state
        is_focused_attr = self.get_attr(element, 'AXFocused')
        is_focused = is_focused_attr is True
        
        # Containers that we should traverse into but not mark as interactive
        is_container = role in {
            'AXWindow', 'AXToolbar', 'AXGroup', 'AXScrollArea', 
            'AXSplitGroup', 'AXList', 'AXTabGroup', 'AXWebArea',
            'AXPopover', 'AXSheet', 'AXLayoutArea', 'AXLayoutItem',
        }

        # Special Case: AXGroup with actions and label should be treated as interactive, not just a container.
        # This is common in SwiftUI where a Group acts as a button.
        if role == 'AXGroup' and has_actions and (title or description or value):
            is_container = False
        
        # Check if this is a scrollable element
        if role in SCROLLABLE_ROLES:
            scroll_node = self._create_scroll_node(
                element, role, title, description, window_name, bbox, is_focused
            )
            if scroll_node:
                scrollable_elements.append(scroll_node)
        
        # Check if this is an interactive element
        # An element is interactive if:
        # 1. It has an interactive role and is enabled, OR
        # 2. It has interactive actions (AXPress, AXConfirm, etc.), OR
        # 3. It has a window control subrole (close, minimize, zoom buttons)
        is_interactive = (
            (role in INTERACTIVE_ROLES and is_enabled) or 
            has_actions or 
            subrole in WINDOW_CONTROL_SUBROLES
        )
        
        # For non-container elements that are interactive
        if not is_container and is_interactive and (title or description or value or has_actions or subrole):
            # Reject interactive elements with 0 width or height
            if w == 0 or h == 0:
                # Element has no visible size, traverse children instead
                for child in self.get_children(element):
                    self.tree_traversal(
                        child, window_name, visited,
                        interactive_elements, scrollable_elements
                    )
                return
            
            bbox = BoundingBox.from_position_size(x, y, w, h)
            center = bbox.get_center()
            
            # Get friendly label for window controls
            friendly_subrole = WINDOW_CONTROL_SUBROLES.get(subrole, subrole or '')
            label = title or description or value or friendly_subrole or "(no label)"
            
            # Get role description (human readable type, e.g. "button", "text field")
            role_description = self.get_attr(element, 'AXRoleDescription')

            node = TreeElementNode(
                bounding_box=bbox,
                center=center,
                role=role or '',
                subrole=subrole or '',
                name=label,
                description=description or '',
                value=value if isinstance(value, str) else '',
                window_name=window_name,
                is_enabled=is_enabled is not False,
                is_focused=is_focused,
                element_type=role_description or '',
                actions=list(actions) if actions else [],
            )
            interactive_elements.append(node)
            
            # If interactive, don't traverse further inside
            return

        # Recurse into children (for containers and non-interactive elements)
        for child in self.get_children(element):
            self.tree_traversal(
                child, window_name, visited,
                interactive_elements, scrollable_elements,
                root_bounds
            )

    def _get_scrollbar_value(self, scrollbar) -> float:
        """
        Get the scroll position value from a scrollbar element.
        Returns a value between 0.0 and 1.0 (0% to 100%).
        """
        if not scrollbar:
            return 0.0
        
        try:
            value = self.get_attr(scrollbar, 'AXValue')
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
        
        return 0.0

    def _create_scroll_node(
        self,
        element,
        role: str,
        title: Optional[str],
        description: Optional[str],
        window_name: str,
        bbox: BoundingBox,
        is_focused: bool
    ) -> Optional[ScrollElementNode]:
        """
        Create a ScrollElementNode from a scrollable element.
        Returns None if the element has no scroll capability.
        """
        # Get horizontal and vertical scrollbars
        h_scrollbar = self.get_attr(element, 'AXHorizontalScrollBar')
        v_scrollbar = self.get_attr(element, 'AXVerticalScrollBar')
        
        # Check if element is actually scrollable
        h_scrollable = h_scrollbar is not None
        v_scrollable = v_scrollbar is not None
        
        # Alternative: check scroll range attributes for elements without explicit scrollbars
        if not h_scrollable and not v_scrollable:
            # Check if there's content that exceeds visible area
            # Some elements use AXVisibleCharacterRange or similar
            visible_rows = self.get_attr(element, 'AXVisibleRows')
            all_rows = self.get_attr(element, 'AXRows')
            if visible_rows and all_rows:
                try:
                    if len(all_rows) > len(visible_rows):
                        v_scrollable = True
                except (TypeError, AttributeError):
                    pass
        
        # If not scrollable at all, don't create a node
        if not h_scrollable and not v_scrollable:
            return None
        
        # Get scroll percentages (0.0 to 1.0 from AXValue, convert to 0-100)
        h_percent = self._get_scrollbar_value(h_scrollbar) * 100.0
        v_percent = self._get_scrollbar_value(v_scrollbar) * 100.0
        
        # Create label
        label = title or description or role or '(scrollable)'
        
        return ScrollElementNode(
            name=label,
            role=role,
            window_name=window_name,
            bounding_box=bbox,
            center=bbox.get_center(),
            horizontal_scrollable=h_scrollable,
            horizontal_scroll_percent=round(h_percent, 1),
            vertical_scrollable=v_scrollable,
            vertical_scroll_percent=round(v_percent, 1),
            is_focused=is_focused,
        )

    def on_focus_changed(self, element):
        """Handle focus change events from WatchDog."""
        # Debounce duplicate events
        import time
        current_time = time.time()
        
        # We can't easily get unique ID like RuntimeId in Windows, 
        # but we can try to get role/title to filter duplicates roughly if needed.
        # For now just log it.
        
        try:
            role = self.get_attr(element, kAXRoleAttribute)
            title = self.get_attr(element, kAXTitleAttribute)
            logger.info(f"[WatchDog] Focus changed to: '{title}' ({role})")
        except Exception:
            pass
