"""
Desktop service for macOS desktop state management and actions.
Provides methods for capturing desktop state, managing applications, and performing input actions.
"""
from macos_use.agent.desktop.views import DesktopState, Window, Size, Status, Browser
from macos_use.agent.desktop.config import BROWSER_BUNDLE_IDS, EXCLUDED_BUNDLE_IDS
from Cocoa import NSWorkspace, NSApplicationActivateIgnoringOtherApps
from macos_use.agent.tree.views import BoundingBox
from markdownify import markdownify as md
from macos_use.agent.tree.service import Tree
from typing import Literal, Optional
from PIL import Image, ImageGrab
from io import BytesIO
import pyautogui as pg
import requests
import subprocess
import Quartz
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

pg.FAILSAFE = False
pg.PAUSE = 0.5


class Desktop:
    """
    Desktop class for macOS desktop automation.
    Manages desktop state capture, application control, and input actions.
    """

    def __init__(self):
        self.tree = Tree(self)
        self.desktop_state: Optional[DesktopState] = None

    def get_screen_size(self) -> Size:
        """Get the combined resolution of all active displays (virtual screen size)."""
        try:
            # Get all active displays
            max_displays = 32
            # Returns (error, display_ids, count)
            res = Quartz.CGGetActiveDisplayList(max_displays, None, None)
            
            if res and res[1]:
                display_ids = res[1]
                
                min_x = float('inf')
                min_y = float('inf')
                max_x = float('-inf')
                max_y = float('-inf')
                
                for display_id in display_ids:
                    bounds = Quartz.CGDisplayBounds(display_id)
                    x = bounds.origin.x
                    y = bounds.origin.y
                    w = bounds.size.width
                    h = bounds.size.height
                    
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x + w)
                    max_y = max(max_y, y + h)
                
                return Size(width=int(max_x - min_x), height=int(max_y - min_y))
                
        except Exception as e:
            logger.warning(f"Failed to calculate virtual screen size: {e}")

        # Fallback to main display
        main_display = Quartz.CGMainDisplayID()
        width = Quartz.CGDisplayPixelsWide(main_display)
        height = Quartz.CGDisplayPixelsHigh(main_display)
        return Size(width=width, height=height)

    def get_macos_version(self) -> str:
        """Get the macOS version."""
        try:
            result = subprocess.run(['sw_vers', '-productVersion'], capture_output=True, text=True)
            version = result.stdout.strip()
            name_result = subprocess.run(['sw_vers', '-productName'], capture_output=True, text=True)
            name = name_result.stdout.strip()
            return f"{name} {version}"
        except Exception:
            return "macOS"

    def get_dpi_scaling(self) -> str:
        """Get the scale factor of the main display."""
        try:
            # On macOS, we can get this from the backing scale factor of the main screen
            # or just assume 2.0 for Retina if we can't get it easily.
            # Using Quartz to find the scale factor.
            main_display = Quartz.CGMainDisplayID()
            # This doesn't directly give scale, but we can check the pixel width vs point width
            pixel_width = Quartz.CGDisplayPixelsWide(main_display)
            bounds = Quartz.CGDisplayBounds(main_display)
            point_width = bounds.size.width
            scale = round(pixel_width / point_width, 1)
            return f"{scale}x"
        except Exception:
            return "1.0x"

    def get_default_language(self) -> str:
        """Get the default system language."""
        try:
            result = subprocess.run(['defaults', 'read', '-g', 'AppleLanguages'], capture_output=True, text=True)
            # Output is like "(en-US, ...)"
            langs = result.stdout.strip()
            if langs.startswith('('):
                first_lang = langs.split(',')[0].strip('() "')
                return first_lang
            return "en-US"
        except Exception:
            return "en-US"

    def get_user_account_type(self) -> str:
        """Check if the current user is an admin."""
        try:
            user = os.getlogin()
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
        # Use Quartz to capture the screen directly (no temp files, no screencapture CLI)
        try:
            # CGWindowListCreateImage captures the full virtual screen
            cg_image = Quartz.CGWindowListCreateImage(
                Quartz.CGRectInfinite,
                Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID,
                Quartz.kCGWindowImageDefault,
            )
            if cg_image is None:
                raise RuntimeError(
                    "CGWindowListCreateImage returned None – "
                    "grant Screen Recording permission in "
                    "System Settings > Privacy & Security > Screen Recording"
                )
            width = Quartz.CGImageGetWidth(cg_image)
            height = Quartz.CGImageGetHeight(cg_image)
            bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
            pixel_data = Quartz.CGDataProviderCopyData(
                Quartz.CGImageGetDataProvider(cg_image)
            )
            img = Image.frombuffer(
                "RGBA", (width, height), pixel_data, "raw", "BGRA", bytes_per_row, 1
            )
        except Exception as e:
            logger.warning(f"Quartz screen capture failed, falling back to ImageGrab: {e}")
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
        from concurrent.futures import ThreadPoolExecutor
        import random
        
        screenshot = self.get_screenshot(as_bytes=False, scale=1.0)
        if screenshot is None:
            return None
            
        # Calculate virtual screen origin and DPI scale factor
        min_x, min_y = 0, 0
        max_logical_x, max_logical_y = 0, 0
        dpi_scale = 1.0
        try:
            max_displays = 32
            # (error, display_ids, count)
            res = Quartz.CGGetActiveDisplayList(max_displays, None, None)
            if res and len(res) > 1:
                display_ids = res[1]
                for display_id in display_ids:
                    bounds = Quartz.CGDisplayBounds(display_id)
                    x = bounds.origin.x
                    y = bounds.origin.y
                    w = bounds.size.width
                    h = bounds.size.height
                    if x < min_x: min_x = int(x)
                    if y < min_y: min_y = int(y)
                    max_logical_x = max(max_logical_x, x + w)
                    max_logical_y = max(max_logical_y, y + h)
                # Compute DPI scale from actual screenshot pixels vs logical screen size
                # This is the most reliable method as CGDisplayPixelsWide can be unreliable
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
            # Try to load a system font
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
            
            # Adjust for virtual screen origin, Retina DPI scale, and padding
            # Accessibility coordinates are in logical points; screenshot is in physical pixels
            left = int((box.left - min_x) * dpi_scale) + padding
            top = int((box.top - min_y) * dpi_scale) + padding
            right = int((box.right - min_x) * dpi_scale) + padding
            bottom = int((box.bottom - min_y) * dpi_scale) + padding
            
            adjusted_box = (left, top, right, bottom)
            
            # Draw bounding box
            draw.rectangle(adjusted_box, outline=color, width=2)
            
            # Label dimensions
            label_width = draw.textlength(str(label), font=font)
            label_height = font_size
            
            # Label position above bounding box
            label_x1 = right - label_width - 4
            label_y1 = top - label_height - 4
            label_x2 = label_x1 + label_width + 4
            label_y2 = label_y1 + label_height + 4
            
            # Keep label within image bounds
            if label_y1 < 0:
                label_y1 = top + 2
                label_y2 = label_y1 + label_height + 4
            
            # Draw label background and text
            draw.rectangle([(label_x1, label_y1), (label_x2, label_y2)], fill=color)
            draw.text((label_x1 + 2, label_y1 + 2), str(label), fill=(255, 255, 255), font=font)
        
        # Draw annotations
        # Using loop instead of ThreadPool for simplicity with shared draw context
        nodes_with_indices = list(enumerate(nodes))
        for i, node in nodes_with_indices:
            draw_annotation(i, node)
        
        # Apply scaling if needed
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
        
        Args:
            use_annotation: If True, draw numbered labels on interactive elements in screenshot.
            use_vision: If True, include a screenshot.
            use_dom: If True, include DOM information for browsers.
            as_bytes: If True, return screenshot as bytes.
            scale: Scale factor for the screenshot.
        
        Returns:
            DesktopState with windows, active window, tree state, and optional screenshot.
        """
        # Get running applications and their windows
        windows = self.get_windows()
        active_window = self.get_active_window(windows)
        
        # Get accessibility tree state
        window_name = active_window.name if active_window else ''
        tree_state = self.tree.get_state(window_name=window_name)
        
        # Capture screenshot if requested
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

    def get_windows(self) -> list[Window]:
        """Get list of user-facing application windows on the desktop."""
        windows = []
        workspace = NSWorkspace.sharedWorkspace()
        
        # Get window list from Quartz - includes on-screen windows
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID
        )
        
        # Create a mapping of PID to window info
        pid_windows = {}
        for win_info in window_list or []:
            pid = win_info.get(Quartz.kCGWindowOwnerPID, 0)
            if pid not in pid_windows:
                pid_windows[pid] = []
            pid_windows[pid].append(win_info)
        
        # Get running applications
        running_apps = workspace.runningApplications()
        
        # NSApplicationActivationPolicy values:
        # 0 = Regular (shows in Dock, user-facing apps)
        # 1 = Accessory (no Dock icon, menu bar apps)
        # 2 = Prohibited (background-only, daemons/services)
        
        for app in running_apps:
            # Only include regular apps (user-facing, show in Dock)
            # activationPolicy() == 0 means NSApplicationActivationPolicyRegular
            if app.activationPolicy() != 0:
                continue
            
            bundle_id = app.bundleIdentifier() or ''
            if bundle_id in EXCLUDED_BUNDLE_IDS:
                continue
            
            app_name = app.localizedName()
            pid = app.processIdentifier()
            is_browser = bundle_id in BROWSER_BUNDLE_IDS
            
            # Check window info from Quartz
            app_windows = pid_windows.get(pid, [])
            
            # Determine status and bounds
            if app.isHidden():
                status = Status.HIDDEN
                bbox = BoundingBox(left=0, top=0, right=0, bottom=0, width=0, height=0)
            elif not app_windows:
                # App is running but has no on-screen windows (minimized or no windows)
                status = Status.MINIMIZED
                bbox = BoundingBox(left=0, top=0, right=0, bottom=0, width=0, height=0)
            else:
                # Get bounds from the first/main window
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
                
                # Check if window is on-screen (layer > 0 typically means normal window)
                layer = main_window.get(Quartz.kCGWindowLayer, 0)
                
                # Check if window is fullscreen (fills screen)
                screen_size = pg.size()
                if width >= screen_size.width and height >= screen_size.height - 50:
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
        
        return windows

    def get_active_window(self, windows: list[Window] = None) -> Optional[Window]:
        """Get the currently active/focused window."""
        workspace = NSWorkspace.sharedWorkspace()
        frontmost = workspace.frontmostApplication()
        
        if not frontmost:
            return None
            
        if windows:
            for window in windows:
                if window.pid == frontmost.processIdentifier():
                    return window
        
        # Create window from frontmost app
        app_name = frontmost.localizedName()
        bundle_id = frontmost.bundleIdentifier() or ''
        pid = frontmost.processIdentifier()
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
        """
        Manage applications: launch, switch to, or resize.
        
        Args:
            mode: 'launch' to open app, 'switch' to focus app, 'resize' to resize window.
            name: Application name for launch/switch modes.
            loc: Window location (x, y) for resize mode.
            size: Window size (width, height) for resize mode.
        
        Returns:
            Status message.
        """
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
        """
        Launch an application by name or bundle ID.
        
        Args:
            name: Application name (e.g., "Safari") or bundle ID (e.g., "com.apple.Safari")
        
        Returns:
            Status message.
        """
        last_error = None
        
        # Check if it looks like a bundle ID (contains dots like com.apple.Safari)
        if '.' in name and not name.endswith('.app'):
            # Try launching by bundle ID
            try:
                subprocess.run(['open', '-b', name], check=True, capture_output=True, text=True)
                return f"Launched {name}"
            except subprocess.CalledProcessError as e:
                last_error = e.stderr.strip() if e.stderr else str(e)
        
        # Try launching by app name using 'open -a' (flexible, case-insensitive)
        try:
            subprocess.run(['open', '-a', name], check=True, capture_output=True, text=True)
            return f"Launched {name}"
        except subprocess.CalledProcessError as e:
            last_error = e.stderr.strip() if e.stderr else str(e)
        
        # Try to find the app using Spotlight (mdfind)
        try:
            result = subprocess.run(
                ['mdfind', f'kMDItemKind == "Application" && kMDItemDisplayName == "*{name}*"cd'],
                capture_output=True, text=True
            )
            apps = result.stdout.strip().split('\n')
            if apps and apps[0]:
                # Launch the first matching app
                subprocess.run(['open', apps[0]], check=True, capture_output=True)
                return f"Launched {apps[0].split('/')[-1].replace('.app', '')}"
        except subprocess.CalledProcessError as e:
            last_error = e.stderr.strip() if e.stderr else str(e)
        
        # Final fallback: NSWorkspace
        workspace = NSWorkspace.sharedWorkspace()
        if workspace.launchApplication_(name):
            return f"Launched {name}"
        
        # App not found - return error
        if last_error and "Unable to find application" in last_error:
            return f"Application '{name}' not found"
        return f"Failed to launch '{name}': Application not found"

    def switch_app(self, name: str) -> str:
        """Switch to an application by name."""
        workspace = NSWorkspace.sharedWorkspace()
        running_apps = workspace.runningApplications()
        
        for app in running_apps:
            if app.localizedName() == name:
                app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
                return f"Switched to {name}"
        
        return f"Application '{name}' not found running"

    def resize_app(self, loc: tuple[int, int] = None, size: tuple[int, int] = None) -> str:
        """Resize the active window (requires Accessibility permissions)."""
        # Note: Full window resize requires using AppleScript or Accessibility API
        # This is a simplified placeholder
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
            try:
                subprocess.run(['osascript', '-e', applescript], check=True, capture_output=True)
                return f"Resized window to loc={loc}, size={size}"
            except subprocess.CalledProcessError as e:
                return f"Failed to resize: {e}"
        
        return "No resize parameters provided"

    def execute_command(self, command: str, mode: str = 'shell', timeout: int = 10) -> tuple[str, int]:
        """
        Execute a command in shell or osascript mode.
        
        Args:
            command: Command to execute.
            mode: 'shell' for bash commands, 'osascript' for AppleScript.
            timeout: Timeout in seconds.
        
        Returns:
            Tuple of (output, return_code).
        """
        env = os.environ.copy()
        try:
            if mode == 'osascript':
                # Execute AppleScript using osascript
                result = subprocess.run(
                    ['osascript', '-e', command],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env
                )
            else:
                # Execute shell command
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env
                )
            output = result.stdout or result.stderr or ''
            return (output.strip(), result.returncode)
        except subprocess.TimeoutExpired:
            return (f"Command timed out after {timeout} seconds", -1)
        except Exception as e:
            return (str(e), -1)

    def click(self, loc: tuple[int, int], button: str = 'left', clicks: int = 1):
        """
        Perform a mouse click at the specified coordinates.
        
        Args:
            loc: (x, y) coordinates.
            button: 'left', 'right', or 'middle'.
            clicks: Number of clicks (0 for hover only, 1 for single, 2 for double).
        """
        x, y = loc
        pg.moveTo(x, y)
        if clicks > 0:
            pg.click(x=x, y=y, button=button, clicks=clicks)

    def type(
        self,
        loc: tuple[int, int],
        text: str,
        caret_position: Literal['start', 'idle', 'end'] = 'idle',
        clear: bool = False,
        press_enter: bool = False
    ):
        """
        Type text at the specified coordinates.
        
        Args:
            loc: (x, y) coordinates to click before typing.
            text: Text to type.
            caret_position: Where to position caret ('start', 'idle', 'end').
            clear: If True, clear existing text before typing.
            press_enter: If True, press Enter after typing.
        """
        x, y = loc
        pg.click(x=x, y=y)
        
        if clear:
            pg.hotkey('command', 'a')
            pg.press('delete')
        
        if caret_position == 'start':
            pg.hotkey('command', 'left')
        elif caret_position == 'end':
            pg.hotkey('command', 'right')
        
        pg.typewrite(text, interval=0.02)
        
        if press_enter:
            pg.press('enter')

    def scroll(
        self,
        loc: tuple[int, int] = None,
        type: Literal['horizontal', 'vertical'] = 'vertical',
        direction: Literal['up', 'down', 'left', 'right'] = 'down',
        wheel_times: int = 1
    ) -> str:
        """
        Scroll at the specified coordinates.
        
        Args:
            loc: (x, y) coordinates. If None, scroll at current position.
            type: 'horizontal' or 'vertical'.
            direction: Scroll direction.
            wheel_times: Number of scroll wheel clicks.
        
        Returns:
            Status message.
        """
        if loc:
            pg.moveTo(loc[0], loc[1])
        
        scroll_amount = wheel_times * 3  # Adjust scroll sensitivity
        
        if type == 'vertical':
            if direction == 'up':
                pg.scroll(scroll_amount)
            else:
                pg.scroll(-scroll_amount)
        else:
            if direction == 'left':
                pg.hscroll(-scroll_amount)
            else:
                pg.hscroll(scroll_amount)
        
        return f"Scrolled {type} {direction}"

    def drag(self, loc: tuple[int, int]):
        """
        Drag from current position to the specified coordinates.
        
        Args:
            loc: Target (x, y) coordinates.
        """
        pg.drag(loc[0] - pg.position()[0], loc[1] - pg.position()[1], duration=0.5)

    def move(self, loc: tuple[int, int]):
        """
        Move the mouse cursor to the specified coordinates.
        
        Args:
            loc: Target (x, y) coordinates.
        """
        pg.moveTo(loc[0], loc[1])

    def shortcut(self, shortcut: str):
        """
        Execute a keyboard shortcut.
        
        Args:
            shortcut: Key combination separated by '+' (e.g., 'command+c', 'command+shift+s').
        """
        keys = shortcut.lower().split('+')
        # Map common Windows keys to macOS
        key_mapping = {
            'ctrl': 'command',
            'win': 'command',
            'alt': 'option',
        }
        mapped_keys = [key_mapping.get(k, k) for k in keys]
        pg.hotkey(*mapped_keys)

    def multi_select(self, press_ctrl: bool = False, locs: list[tuple[int, int]] = None):
        """
        Select multiple items by clicking with Command held.
        
        Args:
            press_ctrl: If True, hold Command key while clicking.
            locs: List of (x, y) coordinates to click.
        """
        if not locs:
            return
        
        if press_ctrl:
            pg.keyDown('command')
        
        for loc in locs:
            pg.click(loc[0], loc[1])
        
        if press_ctrl:
            pg.keyUp('command')

    def multi_edit(self, locs: list[tuple[int, int, str]]):
        """
        Edit multiple fields.
        
        Args:
            locs: List of (x, y, text) tuples.
        """
        for loc in locs:
            x, y, text = loc[0], loc[1], loc[2]
            self.type(loc=(x, y), text=text, clear=True)

    def scrape(self, url: str) -> str:
        """
        Fetch content from a URL and convert to markdown.
        
        Args:
            url: URL to fetch.
        
        Returns:
            Content as markdown text.
        """
        try:            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            html = response.text
            # Convert HTML to markdown
            markdown = md(html, heading_style='ATX', strip=['script', 'style'])
            return markdown.strip()
        except Exception as e:
            return f"Failed to scrape URL: {e}"
