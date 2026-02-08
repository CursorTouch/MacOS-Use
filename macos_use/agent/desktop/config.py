"""
Configuration constants for macOS desktop module.
"""

# Browser bundle identifiers for detection
BROWSER_BUNDLE_IDS = {
    'com.apple.Safari',
    'com.google.Chrome',
    'org.mozilla.firefox',
    'com.microsoft.edgemac',
    'com.brave.Browser',
    'com.operasoftware.Opera',
}

# Application names to avoid (internal/system)
AVOIDED_APPS: set[str] = {
    'Finder',  # Usually always running but not always relevant
}

# System processes to exclude from window listing
EXCLUDED_BUNDLE_IDS: set[str] = {
    "com.apple.loginwindow",
    "com.apple.dock",
    "com.apple.systemuiserver",
    "com.apple.controlcenter",
    "com.apple.notificationcenterui",
    "com.apple.Spotlight",
    "com.apple.ScreenSaver.Engine",
    "com.apple.WindowManager",
    "com.apple.TextInputMenuBar",
    "com.apple.TextInputMenuAgent",
    "com.apple.AirPlayUIAgent",
    "com.apple.PowerChime",
    "com.apple.BezelServices",
}

# Max image dimensions for screenshots
MAX_IMAGE_WIDTH = 1920
MAX_IMAGE_HEIGHT = 1080
