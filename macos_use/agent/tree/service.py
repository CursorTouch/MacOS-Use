"""
Tree service for macOS accessibility tree traversal.
Provides methods to capture and traverse the accessibility tree using the macOS Accessibility API.
Uses multi-threading for parallel traversal of different UI regions.

Now uses the centralized ax module for all accessibility operations.
"""
import macos_use.ax as ax
from macos_use.agent.tree.views import (
    TreeState,
    TreeElementNode,
    ScrollElementNode,
    BoundingBox,
    Center,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, TYPE_CHECKING
from threading import Lock
import weakref
import logging

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

    def has_interactive_actions(self, actions) -> bool:
        """Check if element has interactive actions."""
        if not actions:
            return False
        return any(action in ax.INTERACTIVE_ACTIONS for action in actions)

    def _check_launchpad_and_get_dock(self, apps) -> tuple:
        """
        Check if Launchpad is active and return its element and the dock AX application.
        Returns (launchpad_element, ax_dock).
        launchpad_element is None if Launchpad is not active.
        ax_dock is None if Dock is not running.
        """
        dock_app = next((app for app in apps if app.bundleIdentifier() == "com.apple.dock"), None)
        if not dock_app:
            return None, None
            
        pid = dock_app.processIdentifier()
        ax_dock = ax.ControlFromPID(pid)
        children = ax.GetChildren(ax_dock)
        
        for child in children:
            role = ax.GetAttribute(child, ax.Attribute.Role)
            title = ax.GetAttribute(child, ax.Attribute.Title)
            if role == ax.Role.Group and title == 'Launchpad':
                # Found the Launchpad group. Check if it's visible.
                is_hidden = ax.GetAttribute(child, ax.Attribute.Hidden)
                if is_hidden is not True:
                    return child, ax_dock
                    
        return None, ax_dock

    def is_window_fullscreen(self, window) -> bool:
        """
        Check if a window is in fullscreen mode.
        Uses AXFullScreen attribute to detect fullscreen state.
        """
        if not window:
            return False
        
        try:
            fullscreen = ax.GetAttribute(window, ax.Attribute.FullScreen)
            if fullscreen is True:
                return True
            
            subrole = ax.GetAttribute(window, ax.Attribute.Subrole)
            if subrole == ax.Subrole.FullScreenWindow:
                return True
        except Exception:
            pass
        
        return False

    def get_state(self, window_name: str = '', active_pid: Optional[int] = None) -> TreeState:
        """
        Capture the current accessibility tree state using parallel traversal.
        Returns a TreeState with interactive and scrollable elements.

        Args:
            window_name: Name of the active window (used as fallback label).
            active_pid: PID of the frontmost application as determined by the
                        Desktop service (via CGWindowListCopyWindowInfo). When
                        provided, this is used instead of
                        NSWorkspace.frontmostApplication() which can return
                        stale data when no NSRunLoop is running.
        """
        interactive_elements: list[TreeElementNode] = []
        scrollable_elements: list[ScrollElementNode] = []

        # Get all running apps
        apps = ax.GetRunningApplications()

        # Check if Launchpad is open and get launchpad/dock elements
        launchpad_element, ax_dock = self._check_launchpad_and_get_dock(apps)

        # Prepare tasks for parallel execution
        tasks = []

        if launchpad_element:
            logger.info("Launchpad is active - scanning only Launchpad and Dock")
            
            tasks.append(('launchpad', launchpad_element, 'Launchpad', None))
            
            if ax_dock:
                tasks.append(('dock', ax_dock, 'Dock', None))
        else:
            # Normal mode - scan all sources
            pid = None
            app_name = window_name or ''

            if active_pid:
                pid = active_pid
                for app in apps:
                    if app.processIdentifier() == active_pid:
                        app_name = app.localizedName()
                        break
            else:
                frontmost = ax.GetFrontmostApplication()
                if not frontmost:
                    logger.warning("No frontmost application found")
                    return TreeState()
                pid = frontmost.processIdentifier()
                app_name = frontmost.localizedName()

            ax_app = ax.ControlFromPID(pid)

            # Try to get focused/main window
            focused_window = ax.GetAttribute(ax_app, ax.Attribute.FocusedWindow)
            
            if not focused_window:
                focused_window = ax.GetAttribute(ax_app, ax.Attribute.MainWindow)
                
            if not focused_window:
                windows = ax.GetAttribute(ax_app, ax.Attribute.Windows)
                if windows and len(windows) > 0:
                    focused_window = windows[0]

            window_title = ''
            if focused_window:
                window_title = ax.GetAttribute(focused_window, ax.Attribute.Title) or app_name

            # Task 1: Scan focused window
            if focused_window:
                bounds = None
                pos = ax.GetPosition(focused_window)
                size = ax.GetSize(focused_window)
                if pos and size:
                    bounds = BoundingBox.from_position_size(pos[0], pos[1], size[0], size[1])
                tasks.append(('window', focused_window, window_title, bounds))

            # Task 2: Scan Dock (skip if window is fullscreen/maximized)
            is_fullscreen = self.is_window_fullscreen(focused_window)
            if ax_dock and not is_fullscreen:
                tasks.append(('dock', ax_dock, 'Dock', None))

            # Task 3: Scan Menu Bar of frontmost app
            menu_bar = ax.GetAttribute(ax_app, ax.Attribute.MenuBar)
            if menu_bar:
                tasks.append(('menubar', menu_bar, 'MenuBar', None))

            # Task 4: Scan Control Center (WiFi, Bluetooth, etc.)
            cc_app = next((app for app in apps if app.bundleIdentifier() == "com.apple.controlcenter"), None)
            if cc_app:
                ax_cc = ax.ControlFromPID(cc_app.processIdentifier())
                
                cc_windows = ax.GetAttribute(ax_cc, ax.Attribute.Windows)
                if cc_windows and len(cc_windows) > 0:
                    for cc_window in cc_windows:
                        tasks.append(('control_center_window', cc_window, 'Control Center', None))
                
                cc_extras = ax.GetAttribute(ax_cc, ax.Attribute.ExtrasMenuBar)
                if cc_extras:
                    tasks.append(('control_center', cc_extras, 'Control Center', None))
                
                cc_children = ax.GetChildren(ax_cc)
                for child in cc_children:
                    child_role = ax.GetAttribute(child, ax.Attribute.Role)
                    if child_role in (ax.Role.Popover, ax.Role.Group, ax.Role.Sheet):
                        is_hidden = ax.GetAttribute(child, ax.Attribute.Hidden)
                        if is_hidden is not True:
                            tasks.append(('control_center_popover', child, 'Control Center', None))

            # Task 4.5: Scan Notification Center (Date/Time, Widgets)
            nc_app = next((app for app in apps if app.bundleIdentifier() == "com.apple.notificationcenterui"), None)
            if nc_app:
                ax_nc = ax.ControlFromPID(nc_app.processIdentifier())
                
                nc_windows = ax.GetAttribute(ax_nc, ax.Attribute.Windows)
                if nc_windows:
                    for nc_window in nc_windows:
                        is_hidden = ax.GetAttribute(nc_window, ax.Attribute.Hidden)
                        if is_hidden is not True:
                            tasks.append(('notification_center', nc_window, 'Notification Center', None))
                
                nc_children = ax.GetChildren(ax_nc)
                for child in nc_children:
                    child_role = ax.GetAttribute(child, ax.Attribute.Role)
                    if child_role in (ax.Role.ScrollArea, ax.Role.Group, ax.Role.List):
                        is_hidden = ax.GetAttribute(child, ax.Attribute.Hidden)
                        if is_hidden is not True:
                            tasks.append(('notification_center_child', child, 'Notification Center', None))

            # Task 5: Scan SystemUIServer (battery, volume, etc.)
            ss_app = next((app for app in apps if app.bundleIdentifier() == "com.apple.systemuiserver"), None)
            if ss_app:
                ax_ss = ax.ControlFromPID(ss_app.processIdentifier())
                ss_menu = ax.GetAttribute(ax_ss, ax.Attribute.MenuBar)
                if ss_menu:
                    tasks.append(('system_ui', ss_menu, 'SystemUI', None))

            # Task 6: Scan Spotlight
            sl_app = next((app for app in apps if app.bundleIdentifier() == "com.apple.Spotlight"), None)
            if sl_app:
                ax_sl = ax.ControlFromPID(sl_app.processIdentifier())
                sl_extras = ax.GetAttribute(ax_sl, ax.Attribute.ExtrasMenuBar)
                if sl_extras:
                    tasks.append(('spotlight', sl_extras, 'Spotlight', None))

            # Task 7: Scan Desktop Icons (Finder) - only if no foreground window
            if not focused_window:
                finder_app = next((app for app in apps if app.bundleIdentifier() == "com.apple.finder"), None)
                if finder_app:
                    ax_finder = ax.ControlFromPID(finder_app.processIdentifier())
                    finder_children = ax.GetChildren(ax_finder)
                    if finder_children:
                        for child in finder_children:
                            role = ax.GetAttribute(child, ax.Attribute.Role)
                            desc = ax.GetAttribute(child, ax.Attribute.Description)
                            if role == ax.Role.ScrollArea and desc == 'desktop':
                                tasks.append(('desktop', child, 'Desktop', None))
                                break


        # Execute traversal in parallel
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.get_appwise_nodes, element, name, bounds)
                for _, element, name, bounds in tasks
            ]

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
        is_hidden = ax.GetAttribute(element, ax.Attribute.Hidden)
        if is_hidden is True:
            return

        role = ax.GetAttribute(element, ax.Attribute.Role)
        title = ax.GetAttribute(element, ax.Attribute.Title)
        description = ax.GetAttribute(element, ax.Attribute.Description)
        value = ax.GetAttribute(element, ax.Attribute.Value)
        subrole = ax.GetAttribute(element, ax.Attribute.Subrole)
        is_enabled = ax.GetAttribute(element, ax.Attribute.Enabled)
        actions = ax.GetAttribute(element, ax.Attribute.Actions)

        pos = ax.GetPosition(element)
        size = ax.GetSize(element)

        x, y, w, h = 0, 0, 0, 0
        if pos and size:
            x, y = pos
            w, h = size

        # Check if element has valid geometry
        if w <= 0 or h <= 0:
            for child in ax.GetChildren(element):
                self.tree_traversal(
                    child, window_name, visited,
                    interactive_elements, scrollable_elements,
                    root_bounds
                )
            return

        bbox = BoundingBox.from_position_size(x, y, w, h)
        if root_bounds:
            clipped_bbox = bbox.intersection(root_bounds)
            if not clipped_bbox:
                return
            bbox = clipped_bbox

        has_actions = self.has_interactive_actions(actions)
        
        is_focused_attr = ax.GetAttribute(element, ax.Attribute.Focused)
        is_focused = is_focused_attr is True
        
        is_container = role in ax.CONTAINER_ROLES

        # Special Case: AXGroup with actions and label should be treated as interactive
        if role == ax.Role.Group and has_actions and (title or description or value):
            is_container = False
        
        # Check if this is a scrollable element
        if role in ax.SCROLLABLE_ROLES:
            scroll_node = self._create_scroll_node(
                element, role, title, description, window_name, bbox, is_focused
            )
            if scroll_node:
                scrollable_elements.append(scroll_node)
        
        is_interactive = (
            (role in ax.INTERACTIVE_ROLES and is_enabled is not False) or 
            has_actions or 
            subrole in ax.WINDOW_CONTROL_SUBROLES
        )
        
        if not is_container and is_interactive and (title or description or value or has_actions or subrole):
            if w == 0 or h == 0:
                for child in ax.GetChildren(element):
                    self.tree_traversal(
                        child, window_name, visited,
                        interactive_elements, scrollable_elements
                    )
                return
            
            bbox = BoundingBox.from_position_size(x, y, w, h)
            center = bbox.get_center()
            
            friendly_subrole = ax.WINDOW_CONTROL_SUBROLES.get(subrole, subrole or '')
            label = title or description or value or friendly_subrole or "(no label)"
            
            role_description = ax.GetAttribute(element, ax.Attribute.RoleDescription)

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
            
            return

        # Recurse into children
        for child in ax.GetChildren(element):
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
            value = ax.GetAttribute(scrollbar, ax.Attribute.Value)
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
        h_scrollbar = ax.GetAttribute(element, ax.Attribute.HorizontalScrollBar)
        v_scrollbar = ax.GetAttribute(element, ax.Attribute.VerticalScrollBar)
        
        h_scrollable = h_scrollbar is not None
        v_scrollable = v_scrollbar is not None
        
        if not h_scrollable and not v_scrollable:
            visible_rows = ax.GetAttribute(element, ax.Attribute.VisibleRows)
            all_rows = ax.GetAttribute(element, ax.Attribute.Rows)
            if visible_rows and all_rows:
                try:
                    if len(all_rows) > len(visible_rows):
                        v_scrollable = True
                except (TypeError, AttributeError):
                    pass
        
        if not h_scrollable and not v_scrollable:
            return None
        
        h_percent = self._get_scrollbar_value(h_scrollbar) * 100.0
        v_percent = self._get_scrollbar_value(v_scrollbar) * 100.0
        
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
        import time
        current_time = time.time()
        
        try:
            role = ax.GetAttribute(element, ax.Attribute.Role)
            title = ax.GetAttribute(element, ax.Attribute.Title)
            logger.info(f"[WatchDog] Focus changed to: '{title}' ({role})")
        except Exception:
            pass
