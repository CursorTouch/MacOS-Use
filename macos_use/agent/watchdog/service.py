"""
WatchDog Service for monitoring macOS Accessibility events.
Now uses the centralized ax module's EventObserver for all event observation.

The WatchDog is a thin wrapper around ax.EventObserver that provides the same
interface as before but delegates all AXObserver management to the ax module.
"""
import macos_use.ax as ax
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WatchDog:
    """
    Unified WatchDog Service for monitoring macOS Accessibility events.
    Uses the centralized ax.EventObserver for all observation.
    
    The WatchDog helps overcome laziness in the accessibility tree by:
    1. Monitoring focus changes when users interact with UI
    2. Detecting structure changes when UI elements are added/removed
    3. Tracking property changes like value updates
    
    Usage:
        watchdog = WatchDog()
        watchdog.set_focus_callback(on_focus_change)
        watchdog.set_structure_callback(on_structure_change)
        watchdog.start()
        
        # ... run your app ...
        
        watchdog.stop()
    """
    
    def __init__(self):
        self._observer = ax.EventObserver()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def is_running(self) -> bool:
        """Check if the watchdog is running."""
        return self._observer.is_running
    
    def start(self):
        """Start the watchdog service."""
        self._observer.start()
        logger.debug("WatchDog service started")
    
    def stop(self):
        """Stop the watchdog service."""
        self._observer.stop()
        logger.debug("WatchDog service stopped")
    
    def set_focus_callback(self, callback: Optional[Callable]):
        """
        Set the callback for focus changes. Pass None to disable.
        
        Callback signature: callback(element, notification: str, pid: int)
        - element: The AXUIElement that gained focus
        - notification: The notification type (e.g., 'AXFocusedUIElementChanged')
        - pid: Process ID of the application
        """
        self._observer.on_focus_changed = callback
    
    def set_structure_callback(self, callback: Optional[Callable]):
        """
        Set the callback for structure changes. Pass None to disable.
        
        Callback signature: callback(element, notification: str, pid: int)
        - element: The AXUIElement affected by the structure change
        - notification: The notification type (e.g., 'AXCreated')
        - pid: Process ID of the application
        """
        self._observer.on_structure_changed = callback
    
    def set_property_callback(self, callback: Optional[Callable]):
        """
        Set the callback for property changes. Pass None to disable.
        
        Callback signature: callback(element, notification: str, pid: int)
        - element: The AXUIElement whose property changed
        - notification: The notification type (e.g., 'AXValueChanged')
        - pid: Process ID of the application
        """
        self._observer.on_property_changed = callback
