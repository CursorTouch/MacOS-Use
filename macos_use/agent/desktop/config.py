"""
Configuration constants for macOS desktop module.
"""

# Bundle IDs for known browser applications
BROWSER_BUNDLE_IDS = {
    'com.apple.Safari',
    'com.google.Chrome',
    'org.mozilla.firefox',
    'com.microsoft.edgemac',
    'com.brave.Browser',
    'com.operasoftware.Opera',
    'com.vivaldi.Vivaldi',
    'company.thebrowser.Browser',  # Arc
}

# Bundle IDs for applications to exclude from window listing
EXCLUDED_BUNDLE_IDS = {
    'com.apple.finder',           # Finder (always running, often background)
}

# System UI apps to include in accessibility tree (whitelist).
SYSTEM_UI_BUNDLE_IDS = {
    'com.apple.dock',
    'com.apple.controlcenter',
    'com.apple.systemuiserver',
    'com.apple.Spotlight',
}

# Max image dimensions for screenshots
MAX_IMAGE_WIDTH = 1920
MAX_IMAGE_HEIGHT = 1080
