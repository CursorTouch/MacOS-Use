"""
Desktop service for macOS desktop state management and actions.
Provides methods for capturing desktop state, managing applications, and performing input actions.

Now uses the centralized ax module for accessibility, screen, mouse/keyboard, and window operations.
"""
import macos_use.ax as ax
from macos_use.agent.desktop.views import DesktopState, Window, Size, Status, Browser
from macos_use.agent.desktop.config import BROWSER_BUNDLE_IDS, EXCLUDED_BUNDLE_IDS
from macos_use.agent.tree.views import BoundingBox
from markdownify import markdownify as md
from macos_use.agent.tree.service import Tree
from contextlib import contextmanager
from typing import Literal, Optional
from PIL import Image
from io import BytesIO
import Quartz
import requests
import logging
import time
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Desktop:
    """
    Desktop class for macOS desktop automation.
    Manages desktop state capture, application control, and input actions.
    """

    def __init__(self, use_accessibility: bool = True, use_annotation: bool = True, use_vision: bool = False):
        self.use_accessibility = use_accessibility
        self.use_annotation = use_annotation
        self.use_vision = use_vision
        self.tree = Tree(self)
        self.desktop_state: Optional[DesktopState] = None

    def get_screen_size(self) -> Size:
        """Get the combined resolution of all active displays (virtual screen size)."""
        width, height = ax.GetScreenSize()
        return Size(width=width, height=height)

    def get_macos_version(self) -> str:
        """Get the macOS version."""
        return ax.GetMacOSVersion()

    def get_dpi_scaling(self) -> str:
        """Get the scale factor of the main display."""
        scale = ax.GetDPIScale()
        return f"{scale}x"

    def get_default_language(self) -> str:
        """Get the default system language."""
        return ax.GetDefaultLanguage()

    def get_user_account_type(self) -> str:
        """Check if the current user is an admin."""
        try:
            user = os.getlogin()
            import subprocess
            result = subprocess.run(['dscl', '.', '-read', '/Groups/admin', 'GroupMembership'], capture_output=True, text=True)
            if user in result.stdout:
                return "Admin"
            return "Standard"
        except Exception:
            return "User"

    def get_screenshot(self, as_bytes: bool = False, scale: float = 1.0):
        """
        Capture a screenshot of the desktop (all screens).
        
        Args:
            as_bytes: If True, return image as bytes. Otherwise return PIL Image.
            scale: Scale factor for the screenshot (0.0-1.0).
        
        Returns:
            Screenshot as bytes or PIL Image.
        """
        try:
            cg_image = ax.CaptureScreen()
            if cg_image is None:
                raise RuntimeError(
                    "CGWindowListCreateImage returned None – "
                    "grant Screen Recording permission in "
                    "System Settings > Privacy & Security > Screen Recording"
                )
            img = ax.CGImageToPIL(cg_image)
        except Exception as e:
            logger.warning(f"Quartz screen capture failed, falling back to ImageGrab: {e}")
            from PIL import ImageGrab
            img = ImageGrab.grab(all_screens=True)
        
        # Apply scaling if needed
        if scale < 1.0:
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        if as_bytes:
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            return buffer.getvalue()
            
        return img

    def get_annotated_screenshot(self, nodes: list, scale: float = 1.0) -> Image.Image:
        """
        Capture a screenshot with numbered annotations on interactive elements.
        
        Args:
            nodes: List of TreeElementNode objects to annotate.
            scale: Scale factor for the screenshot.
        
        Returns:
            Annotated PIL Image.
        """
        from PIL import ImageDraw, ImageFont
        import random
        
        screenshot = self.get_screenshot(as_bytes=False, scale=1.0)
        if screenshot is None:
            return None
            
        # Calculate virtual screen origin and DPI scale factor
        min_x, min_y = 0, 0
        max_logical_x, max_logical_y = 0, 0
        dpi_scale = 1.0
        try:
            display_bounds = ax.GetDisplayBounds()
            for bounds in display_bounds:
                if bounds.left < min_x:
                    min_x = int(bounds.left)
                if bounds.top < min_y:
                    min_y = int(bounds.top)
                max_logical_x = max(max_logical_x, bounds.right)
                max_logical_y = max(max_logical_y, bounds.bottom)
            logical_width = max_logical_x - min_x
            if logical_width > 0:
                dpi_scale = screenshot.width / logical_width
        except Exception as e:
            logger.warning(f"Failed to get display bounds: {e}")
        
        # Add padding
        padding = 5
        width = int(screenshot.width + (1.5 * padding))
        height = int(screenshot.height + (1.5 * padding))
        padded_screenshot = Image.new("RGB", (width, height), color=(255, 255, 255))
        padded_screenshot.paste(screenshot, (padding, padding))
        
        draw = ImageDraw.Draw(padded_screenshot)
        font_size = int(12 * dpi_scale)
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
        except IOError:
            try:
                font = ImageFont.truetype('/System/Library/Fonts/SFNSText.ttf', font_size)
            except IOError:
                font = ImageFont.load_default()
        
        def get_random_color():
            return "#{:06x}".format(random.randint(0, 0xFFFFFF))
        
        def draw_annotation(label, node):
            box = node.bounding_box
            color = get_random_color()
            
            left = int((box.left - min_x) * dpi_scale) + padding
            top = int((box.top - min_y) * dpi_scale) + padding
            right = int((box.right - min_x) * dpi_scale) + padding
            bottom = int((box.bottom - min_y) * dpi_scale) + padding
            
            adjusted_box = (left, top, right, bottom)
            
            draw.rectangle(adjusted_box, outline=color, width=2)
            
            label_width = draw.textlength(str(label), font=font)
            label_height = font_size
            
            label_x1 = right - label_width - 4
            label_y1 = top - label_height - 4
            label_x2 = label_x1 + label_width + 4
            label_y2 = label_y1 + label_height + 4
            
            if label_y1 < 0:
                label_y1 = top + 2
                label_y2 = label_y1 + label_height + 4
            
            draw.rectangle([(label_x1, label_y1), (label_x2, label_y2)], fill=color)
            draw.text((label_x1 + 2, label_y1 + 2), str(label), fill=(255, 255, 255), font=font)
        
        nodes_with_indices = list(enumerate(nodes))
        for i, node in nodes_with_indices:
            draw_annotation(i, node)
        
        if scale < 1.0:
            new_width = int(padded_screenshot.width * scale)
            new_height = int(padded_screenshot.height * scale)
            padded_screenshot = padded_screenshot.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return padded_screenshot

    def get_state(
        self,
        use_annotation: bool = True,
        use_vision: bool = False,
        as_bytes: bool = False,
        scale: float = 1.0
    ) -> DesktopState:
        """
        Capture the current desktop state including windows and accessibility tree.
        """
        windows, frontmost_pid = self.get_windows()
        active_window = self.get_active_window(windows, frontmost_pid=frontmost_pid)
        
        active_pid = active_window.pid if active_window else None
        window_name = active_window.name if active_window else ''
        is_browser = active_window.is_browser if active_window else False
        tree_state = self.tree.get_state(window_name=window_name, active_pid=active_pid, is_browser=is_browser)
        
        screenshot = None
        if use_vision:
            if use_annotation:
                nodes = tree_state.interactive_nodes
                screenshot = self.get_annotated_screenshot(nodes=nodes, scale=scale)
            else:
                screenshot = self.get_screenshot(as_bytes=False, scale=scale)
            
            if as_bytes and screenshot:
                buffer = BytesIO()
                screenshot.save(buffer, format='PNG')
                screenshot = buffer.getvalue()
                buffer.close()
        
        self.desktop_state = DesktopState(
            active_window=active_window,
            windows=windows,
            screenshot=screenshot,
            tree_state=tree_state,
        )
        
        return self.desktop_state

    def get_windows(self) -> tuple[list[Window], Optional[int]]:
        """
        Get list of user-facing application windows on the desktop.
        Returns (windows, frontmost_pid).
        """
        windows = []
        
        # Get window list using ax module
        window_list = ax.GetWindowList(on_screen_only=True)
        
        # Identify frontmost PID from z-order
        frontmost_pid = None
        for win_info in window_list:
            layer = win_info.get(Quartz.kCGWindowLayer, -1)
            if layer != 0:
                continue
            pid = win_info.get(Quartz.kCGWindowOwnerPID, 0)
            if pid:
                frontmost_pid = pid
                break

        # Create a mapping of PID to window info
        pid_windows = {}
        for win_info in window_list:
            pid = win_info.get(Quartz.kCGWindowOwnerPID, 0)
            if pid not in pid_windows:
                pid_windows[pid] = []
            pid_windows[pid].append(win_info)
        
        # Get running applications
        running_apps = ax.GetRunningApplications()
        screen_size = ax.GetScreenSize()
        
        for app in running_apps:
            if app.ActivationPolicy != 'Regular':
                continue

            bundle_id = app.BundleIdentifier or ''
            if bundle_id in EXCLUDED_BUNDLE_IDS:
                continue

            app_name = app.LocalizedName
            pid = app.PID
            is_browser = bundle_id in BROWSER_BUNDLE_IDS

            app_windows = pid_windows.get(pid, [])

            if app.IsHidden:
                status = Status.HIDDEN
                bbox = BoundingBox(left=0, top=0, right=0, bottom=0, width=0, height=0)
            elif not app_windows:
                status = Status.MINIMIZED
                bbox = BoundingBox(left=0, top=0, right=0, bottom=0, width=0, height=0)
            else:
                main_window = app_windows[0]
                bounds = main_window.get(Quartz.kCGWindowBounds, {})
                
                x = int(bounds.get('X', 0))
                y = int(bounds.get('Y', 0))
                width = int(bounds.get('Width', 0))
                height = int(bounds.get('Height', 0))
                
                bbox = BoundingBox(
                    left=x,
                    top=y,
                    right=x + width,
                    bottom=y + height,
                    width=width,
                    height=height
                )
                
                if width >= screen_size[0] and height >= screen_size[1] - 50:
                    status = Status.FULL_SCREEN
                else:
                    status = Status.NORMAL
            
            windows.append(Window(
                name=app_name,
                is_browser=is_browser,
                status=status,
                bounding_box=bbox,
                pid=pid,
                bundle_id=bundle_id,
            ))
        
        return windows, frontmost_pid

    def _get_frontmost_pid(self) -> Optional[int]:
        """Get the PID of the frontmost application using CGWindowListCopyWindowInfo."""
        return ax.GetForegroundWindowPID()

    def get_active_window(self, windows: list[Window] = None, frontmost_pid: Optional[int] = None) -> Optional[Window]:
        """Get the currently active/focused window."""
        if frontmost_pid is None:
            frontmost_pid = self._get_frontmost_pid()

        if frontmost_pid and windows:
            for window in windows:
                if window.pid == frontmost_pid:
                    return window

        frontmost = ax.GetFrontmostApplication()
        if not frontmost:
            return None

        if windows:
            for window in windows:
                if window.pid == frontmost.PID:
                    return window

        app_name = frontmost.LocalizedName
        bundle_id = frontmost.BundleIdentifier or ''
        pid = frontmost.PID
        is_browser = bundle_id in BROWSER_BUNDLE_IDS

        return Window(
            name=app_name,
            is_browser=is_browser,
            status=Status.NORMAL,
            bounding_box=BoundingBox(left=0, top=0, right=0, bottom=0, width=0, height=0),
            pid=pid,
            bundle_id=bundle_id,
        )

    def app(
        self,
        mode: Literal['launch', 'switch', 'resize']='launch',
        name: str = None,
        loc: Optional[tuple[int, int]] = None,
        size: Optional[tuple[int, int]] = None
    ) -> str:
        """Manage applications: launch, switch to, or resize."""
        match mode:
            case 'launch':
                if not name:
                    return "Error: Application name required for launch mode"
                return self.launch_app(name)
            case 'switch':
                if not name:
                    return "Error: Application name required for switch mode"
                return self.switch_app(name)
            case 'resize':
                return self.resize_app(loc=loc, size=size)
            case _:
                return f"Error: Unknown mode '{mode}'"

    def launch_app(self, name: str) -> str:
        """Launch an application by name or bundle ID."""
        import subprocess
        last_error = None
        
        if '.' in name and not name.endswith('.app'):
            try:
                subprocess.run(['open', '-b', name], check=True, capture_output=True, text=True)
                return f"Launched {name}"
            except subprocess.CalledProcessError as e:
                last_error = e.stderr.strip() if e.stderr else str(e)
        
        if ax.LaunchApplication(name):
            return f"Launched {name}"
        
        # Try to find the app using Spotlight (mdfind)
        try:
            result = subprocess.run(
                ['mdfind', f'kMDItemKind == "Application" && kMDItemDisplayName == "*{name}*"cd'],
                capture_output=True, text=True
            )
            apps = result.stdout.strip().split('\n')
            if apps and apps[0]:
                subprocess.run(['open', apps[0]], check=True, capture_output=True)
                return f"Launched {apps[0].split('/')[-1].replace('.app', '')}"
        except subprocess.CalledProcessError as e:
            last_error = e.stderr.strip() if e.stderr else str(e)
        
        if last_error and "Unable to find application" in last_error:
            return f"Application '{name}' not found"
        return f"Failed to launch '{name}': Application not found"

    def _resolve_app_pid(self, name: str) -> Optional[int]:
        """Resolve an application name to its PID (if currently running)."""
        for app in ax.GetRunningApplications():
            if app.LocalizedName == name:
                return app.PID
        return None

    def _wait_for_app_focus(self, target_pid: int, timeout: float = 3.0, interval: float = 0.2) -> bool:
        """Poll the window server until target_pid owns the topmost window."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frontmost_pid = ax.GetForegroundWindowPID()
            if frontmost_pid == target_pid:
                return True
            time.sleep(interval)
        return False

    def switch_app(self, name: str) -> str:
        """Switch to an application by name."""
        target_pid = self._resolve_app_pid(name)

        applescript = f'tell application "{name}" to activate'
        try:
            import subprocess
            subprocess.run(['osascript', '-e', applescript], check=True, capture_output=True)
        except Exception:
            if target_pid is not None:
                if ax.ActivateApplication(target_pid):
                    pass
                else:
                    return f"Application '{name}' not found running"
            else:
                return f"Application '{name}' not found running"

        if target_pid is not None:
            if self._wait_for_app_focus(target_pid):
                return f"Switched to {name}"
            else:
                logger.warning(f"Timed out waiting for '{name}' (PID {target_pid}) to become frontmost")
                return f"Switched to {name} (focus may be delayed)"
        
        time.sleep(0.5)
        return f"Switched to {name}"

    def resize_app(self, loc: tuple[int, int] = None, size: tuple[int, int] = None) -> str:
        """Resize the active window (requires Accessibility permissions)."""
        if loc or size:
            script = []
            if loc:
                script.append(f'set position of front window to {{{loc[0]}, {loc[1]}}}')
            if size:
                script.append(f'set size of front window to {{{size[0]}, {size[1]}}}')
            
            applescript = f'''
            tell application "System Events"
                tell (first process whose frontmost is true)
                    {chr(10).join(script)}
                end tell
            end tell
            '''
            response, status = ax.ExecuteCommand(applescript, mode='osascript', timeout=15)
            if status == 0:
                return f"Resized window to loc={loc}, size={size}"
            return f"Failed to resize: {response}"
        
        return "No resize parameters provided"

    def execute_command(self, command: str, mode: str = 'shell', timeout: int = 10) -> tuple[str, int]:
        """Execute a command in shell or osascript mode."""
        return ax.ExecuteCommand(command, mode=mode, timeout=timeout)

    def click(self, loc: tuple[int, int], button: str = 'left', clicks: int = 1):
        """Perform a mouse click at the specified coordinates."""
        x, y = loc
        ax.MoveTo(x, y)
        if clicks > 0:
            time.sleep(0.05)
            if button == 'right':
                ax.RightClick(x, y)
            elif button == 'middle':
                ax.MiddleClick(x, y)
            elif clicks == 2:
                ax.DoubleClick(x, y)
            else:
                for _ in range(clicks):
                    ax.Click(x, y)

    @staticmethod
    def _is_truthy(value) -> bool:
        """Convert a value to bool, handling string 'true'/'false' from LLM tool calls."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() == 'true'
        return bool(value)

    def type(
        self,
        loc: tuple[int, int],
        text: str,
        caret_position: Literal['start', 'idle', 'end'] = 'idle',
        clear: bool = False,
        press_enter: bool = False
    ):
        """Type text at the specified coordinates."""
        clear = self._is_truthy(clear)
        press_enter = self._is_truthy(press_enter)

        x, y = loc
        ax.Click(x, y)
        time.sleep(0.05)
        
        if clear:
            ax.HotKey('command', 'a')
            time.sleep(0.05)
            ax.KeyPress(ax.KeyCode.Delete)
        
        if caret_position == 'start':
            ax.HotKey('command', 'left')
            time.sleep(0.05)
        elif caret_position == 'end':
            ax.HotKey('command', 'right')
            time.sleep(0.05)
        
        ax.TypeText(text)
        
        if press_enter:
            time.sleep(0.05)
            ax.KeyPress(ax.KeyCode.Return)

    def scroll(
        self,
        loc: tuple[int, int] = None,
        type: Literal['horizontal', 'vertical'] = 'vertical',
        direction: Literal['up', 'down', 'left', 'right'] = 'down',
        wheel_times: int = 1
    ) -> str:
        """Scroll at the specified coordinates."""
        if loc:
            ax.MoveTo(loc[0], loc[1])
        
        if type == 'vertical':
            if direction == 'up':
                ax.WheelUp(clicks=wheel_times)
            else:
                ax.WheelDown(clicks=wheel_times)
        else:
            if direction == 'left':
                ax.WheelLeft(clicks=wheel_times)
            else:
                ax.WheelRight(clicks=wheel_times)
        
        return f"Scrolled {type} {direction}"

    def drag(self, loc: tuple[int, int]):
        """Drag from current position to the specified coordinates."""
        current = ax.GetCursorPos()
        ax.DragTo(current[0], current[1], loc[0], loc[1])

    def move(self, loc: tuple[int, int]):
        """Move the mouse cursor to the specified coordinates."""
        ax.MoveTo(loc[0], loc[1])

    def shortcut(self, shortcut: str):
        """Execute a keyboard shortcut."""
        keys = shortcut.lower().split('+')
        # Map common Windows and alias keys to macOS
        key_mapping = {
            'ctrl': 'command',
            'cmd': 'command',
            'command': 'command',
            'win': 'command',
            'alt': 'option',
            'opt': 'option',
        }
        mapped_keys = [key_mapping.get(k.strip(), k.strip()) for k in keys]
        ax.HotKey(*mapped_keys)

    def multi_select(self, press_ctrl: bool = False, locs: list[tuple[int, int]] = None):
        """Select multiple items by clicking with Command held."""
        if not locs:
            return
        
        if press_ctrl:
            ax.KeyDown(ax.KeyCode.Command)
        
        for loc in locs:
            ax.Click(loc[0], loc[1])
        
        if press_ctrl:
            ax.KeyUp(ax.KeyCode.Command)

    def multi_edit(self, locs: list[tuple[int, int, str]]):
        """Edit multiple fields."""
        for loc in locs:
            x, y, text = loc[0], loc[1], loc[2]
            self.type(loc=(x, y), text=text, clear=True)

    def manage_spaces(
        self,
        action: str,
        desktop_name: str = None,
        new_name: str = None,
    ) -> str:
        """Manage macOS virtual desktops (Spaces) via Mission Control."""
        match action:
            case 'create':
                create_script = '\n'.join([
                    'tell application "System Events"',
                    '    key code 126 using {control down}',
                    '    delay 1.0',
                    'end tell',
                    '',
                    'tell application "System Events" to tell process "Dock"',
                    '    click button 1 of group 2 of group 1 of group 1',
                    'end tell',
                    '',
                    'delay 0.5',
                    '',
                    'tell application "System Events"',
                    '    key code 53',
                    'end tell',
                ])
                response, status = ax.ExecuteCommand(create_script, mode='osascript', timeout=15)
                if status == 0:
                    return "Created a new Space via Mission Control."
                return f"Error creating Space: {response}"

            case 'remove':
                if not desktop_name:
                    remove_script = '\n'.join([
                        'tell application "System Events"',
                        '    key code 126 using {control down}',
                        '    delay 1.0',
                        'end tell',
                        '',
                        'tell application "System Events" to tell process "Dock"',
                        '    set mcGroup to group 1 of group 1',
                        '    set spacesBar to list 1 of group 2 of mcGroup',
                        '    set allSpaces to buttons of spacesBar',
                        '    if (count of allSpaces) is less than or equal to 1 then',
                        '        error "Cannot remove the last remaining Space."',
                        '    end if',
                        '    repeat with sp in allSpaces',
                        '        if value of attribute "AXIsSelected" of sp is true then',
                        '            perform action "AXRemoveDesktop" of sp',
                        '            exit repeat',
                        '        end if',
                        '    end repeat',
                        'end tell',
                        '',
                        'delay 0.5',
                        '',
                        'tell application "System Events"',
                        '    key code 53',
                        'end tell',
                    ])
                    response, status = ax.ExecuteCommand(remove_script, mode='osascript', timeout=15)
                    if status == 0:
                        return "Removed the current Space."
                    return f"Error removing Space: {response}"
                else:
                    return ("Error: To remove a specific Space, first switch to it "
                            "using the 'switch' action, then call remove without desktop_name.")

            case 'rename':
                return "Error: Renaming Spaces is not supported on macOS. Spaces are identified by their number."

            case 'switch':
                if not desktop_name:
                    return ("Error: desktop_name is required for switching. "
                            "Use a number (e.g., '1', '2') or direction ('left', 'right', 'next', 'previous').")

                name = desktop_name.strip().lower()

                if name in ('left', 'previous'):
                    script = 'tell application "System Events" to key code 123 using {control down}'
                    response, status = ax.ExecuteCommand(script, mode='osascript')
                    if status == 0:
                        return "Switched to the Space on the left."
                    return f"Error switching Space: {response}"

                elif name in ('right', 'next'):
                    script = 'tell application "System Events" to key code 124 using {control down}'
                    response, status = ax.ExecuteCommand(script, mode='osascript')
                    if status == 0:
                        return "Switched to the Space on the right."
                    return f"Error switching Space: {response}"

                elif name.isdigit():
                    space_num = int(name)
                    if space_num < 1 or space_num > 9:
                        return f"Error: Space number must be between 1 and 9. Got: {space_num}"
                    key_codes = {1: 18, 2: 19, 3: 20, 4: 21, 5: 23, 6: 22, 7: 26, 8: 28, 9: 25}
                    key_code = key_codes[space_num]
                    script = f'tell application "System Events" to key code {key_code} using {{control down}}'
                    response, status = ax.ExecuteCommand(script, mode='osascript')
                    if status == 0:
                        return f"Switched to Space {space_num}."
                    return (
                        f"Error switching to Space {space_num}: {response}. "
                        f"Ensure 'Switch to Desktop {space_num}' is enabled in "
                        f"System Settings > Keyboard > Shortcuts > Mission Control."
                    )

                else:
                    return (f"Error: Invalid desktop_name '{desktop_name}'. "
                            "Use a number (1-9) or direction ('left', 'right', 'next', 'previous').")

            case _:
                return f"Error: Unknown action: {action}"

    @contextmanager
    def auto_minimize(self):
        """
        Context manager that minimizes the foreground window on entry
        and restores it on exit. Used to keep the agent's terminal/interface
        out of the way while interacting with other windows.
        """
        window_element = None
        try:
            pid = ax.GetForegroundWindowPID()
            if pid:
                app_element = ax.ControlFromPID(pid)
                if app_element:
                    control = ax.Control(element=app_element)
                    # Try focused window first, then main window
                    window_control = control.FocusedWindow or control.MainWindow
                    if window_control and not window_control.IsMinimized:
                        window_element = window_control.Element
                        ax.SetAttribute(window_element, ax.Attribute.Minimized, True)
                        logger.info(f"[Desktop] Auto-minimized foreground window (PID: {pid})")
                        time.sleep(0.3)  # Brief pause for the animation
            yield
        finally:
            if window_element:
                try:
                    ax.SetAttribute(window_element, ax.Attribute.Minimized, False)
                    logger.info(f"[Desktop] Restored minimized window")
                except Exception as e:
                    logger.warning(f"[Desktop] Failed to restore window: {e}")

    def scrape(self, url: str) -> str:
        """Fetch content from a URL and convert to markdown."""
        try:            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            html = response.text
            markdown = md(html, heading_style='ATX', strip=['script', 'style'])
            return markdown.strip()
        except Exception as e:
            return f"Failed to scrape URL: {e}"
