"""
WatchDog Service for monitoring macOS Accessibility events.
Uses AXObserver to track focus changes and UI structure changes.
"""
from ApplicationServices import (
    AXObserverCreate,
    AXObserverAddNotification,
    AXObserverRemoveNotification,
    AXObserverGetRunLoopSource,
    AXUIElementCreateApplication,
    AXUIElementCopyAttributeValue,
    kAXFocusedUIElementChangedNotification,
    kAXWindowCreatedNotification,
    kAXUIElementDestroyedNotification,
    kAXSelectedChildrenChangedNotification,
    kAXRowCountChangedNotification,
    kAXSelectedRowsChangedNotification,
    kAXValueChangedNotification,
    kAXTitleChangedNotification,
    kAXSelectedTextChangedNotification,
    kAXMenuOpenedNotification,
    kAXMenuClosedNotification,
    kAXFocusedWindowChangedNotification,
    kAXCreatedNotification,
    kAXMovedNotification,
    kAXResizedNotification,
    kAXErrorSuccess,
    kAXRoleAttribute,
    kAXTitleAttribute,
)
from Cocoa import NSWorkspace
import objc
from CoreFoundation import (
    CFRunLoopGetCurrent,
    CFRunLoopAddSource,
    CFRunLoopRemoveSource,
    CFRunLoopRunInMode,
    kCFRunLoopDefaultMode,
)
from threading import Thread, Event, Lock
from typing import Callable, Optional, Set, Any
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Notification types for structure changes
STRUCTURE_NOTIFICATIONS = {
    str(kAXCreatedNotification),
    str(kAXUIElementDestroyedNotification),
    str(kAXWindowCreatedNotification),
    str(kAXMenuOpenedNotification),
    str(kAXMenuClosedNotification),
    str(kAXRowCountChangedNotification),
}

# Notification types for focus changes
FOCUS_NOTIFICATIONS = {
    str(kAXFocusedUIElementChangedNotification),
    str(kAXFocusedWindowChangedNotification),
}

# Notification types for property/value changes
PROPERTY_NOTIFICATIONS = {
    str(kAXValueChangedNotification),
    str(kAXTitleChangedNotification),
    str(kAXSelectedTextChangedNotification),
    str(kAXSelectedChildrenChangedNotification),
    str(kAXSelectedRowsChangedNotification),
    str(kAXMovedNotification),
    str(kAXResizedNotification),
}

# All notification constants as a list for registration
ALL_NOTIFICATIONS = [
    kAXFocusedUIElementChangedNotification,
    kAXFocusedWindowChangedNotification,
    kAXCreatedNotification,
    kAXUIElementDestroyedNotification,
    kAXWindowCreatedNotification,
    kAXMenuOpenedNotification,
    kAXMenuClosedNotification,
    kAXRowCountChangedNotification,
    kAXValueChangedNotification,
    kAXTitleChangedNotification,
    kAXSelectedTextChangedNotification,
    kAXSelectedChildrenChangedNotification,
    kAXSelectedRowsChangedNotification,
    kAXMovedNotification,
    kAXResizedNotification,
]


# Global registry for WatchDog instances (needed for callback routing)
_watchdog_registry: dict[int, 'WatchDog'] = {}


@objc.callbackFor(AXObserverCreate)
def _global_observer_callback(observer, element, notification, refcon):
    """
    Global callback function for AXObserver notifications.
    Routes notifications to appropriate WatchDog instances.
    
    Args:
        observer: The AXObserver that received the notification.
        element: The AXUIElement that generated the notification.
        notification: The notification name (CFString).
        refcon: Reference constant (PID of the observed app).
    """
    try:
        # Find the WatchDog instance that owns this observer
        for watchdog in _watchdog_registry.values():
            if watchdog._has_observer_for_element(observer):
                watchdog._dispatch_notification(element, str(notification), refcon)
                break
    except Exception as e:
        logger.debug(f"Error in global observer callback: {e}")


class AppObserver:
    """
    Observer for a single application's accessibility events.
    Each application requires its own AXObserver instance.
    """
    
    def __init__(self, pid: int, watchdog: 'WatchDog'):
        self.pid = pid
        self.watchdog = watchdog
        self.observer = None
        self.ax_app = None
        self.run_loop_source = None
        self.registered_notifications: Set[str] = set()
        
    def start(self, notifications: list) -> bool:
        """
        Create the observer and register for notifications.
        
        Args:
            notifications: List of notification constants to watch for.
            
        Returns:
            True if successfully started, False otherwise.
        """
        try:
            # Create AXUIElement for the application
            self.ax_app = AXUIElementCreateApplication(self.pid)
            if not self.ax_app:
                logger.debug(f"Failed to create AXUIElement for PID {self.pid}")
                return False
            
            # Create observer with the global callback
            # Pass PID as refcon for identification
            error, self.observer = AXObserverCreate(
                self.pid, 
                _global_observer_callback, 
                None
            )
            if error != kAXErrorSuccess or not self.observer:
                logger.debug(f"Failed to create AXObserver for PID {self.pid}: error {error}")
                return False
            
            # Get run loop source
            self.run_loop_source = AXObserverGetRunLoopSource(self.observer)
            if not self.run_loop_source:
                logger.debug(f"Failed to get run loop source for PID {self.pid}")
                return False
            
            # Add to run loop
            run_loop = CFRunLoopGetCurrent()
            CFRunLoopAddSource(run_loop, self.run_loop_source, kCFRunLoopDefaultMode)
            
            # Register for each notification type
            for notification in notifications:
                try:
                    error = AXObserverAddNotification(
                        self.observer,
                        self.ax_app,
                        notification,
                        self.pid  # Pass PID as refcon
                    )
                    if error == kAXErrorSuccess:
                        self.registered_notifications.add(str(notification))
                    else:
                        logger.debug(f"Failed to add notification {notification} for PID {self.pid}: error {error}")
                except Exception as e:
                    logger.debug(f"Exception adding notification {notification}: {e}")
            
            logger.debug(f"Started observer for PID {self.pid} with {len(self.registered_notifications)} notifications")
            return len(self.registered_notifications) > 0
            
        except Exception as e:
            logger.debug(f"Exception starting observer for PID {self.pid}: {e}")
            return False
    
    def stop(self):
        """Stop the observer and clean up resources."""
        try:
            # Remove all registered notifications
            if self.observer and self.ax_app:
                for notification_str in list(self.registered_notifications):
                    try:
                        # Find the matching notification constant
                        for notif in ALL_NOTIFICATIONS:
                            if str(notif) == notification_str:
                                AXObserverRemoveNotification(
                                    self.observer,
                                    self.ax_app,
                                    notif
                                )
                                break
                    except Exception:
                        pass
                self.registered_notifications.clear()
            
            # Remove from run loop
            if self.run_loop_source:
                try:
                    CFRunLoopRemoveSource(
                        CFRunLoopGetCurrent(),
                        self.run_loop_source,
                        kCFRunLoopDefaultMode
                    )
                except Exception:
                    pass
                self.run_loop_source = None
            
            self.observer = None
            self.ax_app = None
            logger.debug(f"Stopped observer for PID {self.pid}")
            
        except Exception as e:
            logger.debug(f"Exception stopping observer for PID {self.pid}: {e}")
    
    def matches_observer(self, observer) -> bool:
        """Check if this AppObserver owns the given observer."""
        return self.observer is observer


class WatchDog:
    """
    Unified WatchDog Service for monitoring macOS Accessibility events.
    Manages multiple AppObserver instances to track changes across applications.
    
    The WatchDog helps overcome laziness in the accessibility tree by:
    1. Monitoring focus changes when users interact with UI
    2. Detecting structure changes when UI elements are added/removed
    3. Tracking property changes like value updates
    
    This allows the tree traversal to be more complete as notifications
    trigger callbacks that can force fresh tree reads.
    
    Usage:
        watchdog = WatchDog()
        watchdog.set_focus_callback(on_focus_change)
        watchdog.set_structure_callback(on_structure_change)
        watchdog.start()
        
        # ... run your app ...
        
        watchdog.stop()
    """
    
    _instance_counter = 0
    
    def __init__(self):
        WatchDog._instance_counter += 1
        self._instance_id = WatchDog._instance_counter
        
        self.is_running = Event()
        self.thread: Optional[Thread] = None
        self._lock = Lock()
        
        # Callbacks
        self._focus_callback: Optional[Callable] = None
        self._structure_callback: Optional[Callable] = None
        self._property_callback: Optional[Callable] = None
        
        # Track observed applications
        self._app_observers: dict[int, AppObserver] = {}
        self._observed_pids: Set[int] = set()
        
        # Debouncing
        self._last_event_time: float = 0
        self._debounce_interval: float = 0.05  # 50ms debounce
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def start(self):
        """Start the watchdog service thread."""
        if self.is_running.is_set():
            return
        
        # Register this instance
        _watchdog_registry[self._instance_id] = self
        
        self.is_running.set()
        self.thread = Thread(target=self._run, name=f"WatchDogThread-{self._instance_id}")
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"WatchDog service started (instance {self._instance_id})")
    
    def stop(self):
        """Stop the watchdog service thread."""
        if not self.is_running.is_set():
            return
        
        self.is_running.clear()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        # Unregister this instance
        _watchdog_registry.pop(self._instance_id, None)
        
        logger.info(f"WatchDog service stopped (instance {self._instance_id})")
    
    def set_focus_callback(self, callback: Optional[Callable]):
        """
        Set the callback for focus changes. Pass None to disable.
        
        Callback signature: callback(element, notification: str, pid: int)
        - element: The AXUIElement that gained focus
        - notification: The notification type (e.g., 'AXFocusedUIElementChanged')
        - pid: Process ID of the application
        """
        self._focus_callback = callback
    
    def set_structure_callback(self, callback: Optional[Callable]):
        """
        Set the callback for structure changes. Pass None to disable.
        
        Callback signature: callback(element, notification: str, pid: int)
        - element: The AXUIElement affected by the structure change
        - notification: The notification type (e.g., 'AXCreated')
        - pid: Process ID of the application
        """
        self._structure_callback = callback
    
    def set_property_callback(self, callback: Optional[Callable]):
        """
        Set the callback for property changes. Pass None to disable.
        
        Callback signature: callback(element, notification: str, pid: int)
        - element: The AXUIElement whose property changed
        - notification: The notification type (e.g., 'AXValueChanged')
        - pid: Process ID of the application
        """
        self._property_callback = callback
    
    def _has_observer_for_element(self, observer) -> bool:
        """Check if this WatchDog owns the given observer."""
        with self._lock:
            for app_observer in self._app_observers.values():
                if app_observer.matches_observer(observer):
                    return True
        return False
    
    def _dispatch_notification(self, element, notification: str, pid: int):
        """
        Dispatch a notification to the appropriate callback.
        
        Args:
            element: The AXUIElement that generated the notification.
            notification: The notification name.
            pid: The process ID of the application.
        """
        # Debounce rapid events
        current_time = time.time()
        if current_time - self._last_event_time < self._debounce_interval:
            return
        self._last_event_time = current_time
        
        try:
            # Route to appropriate callback based on notification type
            if notification in FOCUS_NOTIFICATIONS and self._focus_callback:
                self._focus_callback(element, notification, pid)
            elif notification in STRUCTURE_NOTIFICATIONS and self._structure_callback:
                self._structure_callback(element, notification, pid)
            elif notification in PROPERTY_NOTIFICATIONS and self._property_callback:
                self._property_callback(element, notification, pid)
        except Exception as e:
            logger.debug(f"Error in notification dispatch: {e}")
    
    def _get_running_app_pids(self) -> Set[int]:
        """Get PIDs of all running user-facing applications."""
        pids = set()
        try:
            apps = NSWorkspace.sharedWorkspace().runningApplications()
            for app in apps:
                # Only include regular apps (activationPolicy == 0)
                if app.activationPolicy() == 0:
                    pids.add(app.processIdentifier())
        except Exception as e:
            logger.debug(f"Error getting running app PIDs: {e}")
        return pids
    
    def _get_notifications_to_register(self) -> list:
        """Determine which notification constants to register based on set callbacks."""
        notifications = []
        
        if self._focus_callback:
            notifications.extend([
                kAXFocusedUIElementChangedNotification,
                kAXFocusedWindowChangedNotification,
            ])
        
        if self._structure_callback:
            notifications.extend([
                kAXCreatedNotification,
                kAXUIElementDestroyedNotification,
                kAXWindowCreatedNotification,
                kAXMenuOpenedNotification,
                kAXMenuClosedNotification,
                kAXRowCountChangedNotification,
            ])
        
        if self._property_callback:
            notifications.extend([
                kAXValueChangedNotification,
                kAXTitleChangedNotification,
                kAXSelectedTextChangedNotification,
                kAXSelectedChildrenChangedNotification,
                kAXSelectedRowsChangedNotification,
                kAXMovedNotification,
                kAXResizedNotification,
            ])
        
        return notifications
    
    def _update_observers(self):
        """
        Update observers to match currently running applications.
        Adds observers for new apps, removes observers for terminated apps.
        """
        with self._lock:
            current_pids = self._get_running_app_pids()
            notifications = self._get_notifications_to_register()
            
            if not notifications:
                # No callbacks set, remove all observers
                for pid, observer in list(self._app_observers.items()):
                    observer.stop()
                self._app_observers.clear()
                self._observed_pids.clear()
                return
            
            # Remove observers for apps that are no longer running
            terminated_pids = self._observed_pids - current_pids
            for pid in terminated_pids:
                if pid in self._app_observers:
                    self._app_observers[pid].stop()
                    del self._app_observers[pid]
                self._observed_pids.discard(pid)
            
            # Add observers for new apps
            new_pids = current_pids - self._observed_pids
            for pid in new_pids:
                observer = AppObserver(pid, self)
                if observer.start(notifications):
                    self._app_observers[pid] = observer
                    self._observed_pids.add(pid)
    
    def _run(self):
        """Main event loop running in a dedicated thread."""
        try:
            while self.is_running.is_set():
                # Update observers to track running applications
                self._update_observers()
                
                # Run the run loop for a short interval to process events
                # This is the macOS equivalent of Windows COM PumpEvents
                CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.1, False)
                
        except Exception as e:
            logger.error(f"WatchDog died: {e}")
        finally:
            # Cleanup all observers on exit
            with self._lock:
                for observer in self._app_observers.values():
                    try:
                        observer.stop()
                    except Exception:
                        pass
                self._app_observers.clear()
                self._observed_pids.clear()
            logger.info(f"WatchDog thread exited (instance {self._instance_id})")