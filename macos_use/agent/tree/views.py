"""
Data classes for representing macOS accessibility tree elements and state.
"""
from dataclasses import dataclass, field
from typing import Optional, Any
import json

WARNING_MESSAGE="The desktop UI services are temporarily unavailable. Please wait a few seconds and continue."
EMPTY_MESSAGE="No elements found"

@dataclass
class BoundingBox:
    """Represents the bounding box of a UI element."""
    left: float
    top: float
    right: float
    bottom: float
    width: float
    height: float

    @classmethod
    def from_position_size(cls, x: float, y: float, w: float, h: float) -> 'BoundingBox':
        """Create a BoundingBox from position and size."""
        return cls(
            left=x,
            top=y,
            right=x + w,
            bottom=y + h,
            width=w,
            height=h
        )

    def get_center(self) -> 'Center':
        """Get the center point of the bounding box."""
        return Center(
            x=int(self.left + self.width / 2),
            y=int(self.top + self.height / 2)
        )

    def xywh_to_string(self) -> str:
        return f'({int(self.left)},{int(self.top)},{int(self.width)},{int(self.height)})'
    
    def xyxy_to_string(self) -> str:
        return f'({int(self.left)},{int(self.top)},{int(self.right)},{int(self.bottom)})'

    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box intersects with another."""
        return not (
            self.right < other.left or 
            self.left > other.right or 
            self.bottom < other.top or 
            self.top > other.bottom
        )

    def intersection(self, other: 'BoundingBox') -> Optional['BoundingBox']:
        """Calculate the intersection of two bounding boxes."""
        new_left = max(self.left, other.left)
        new_top = max(self.top, other.top)
        new_right = min(self.right, other.right)
        new_bottom = min(self.bottom, other.bottom)

        if new_left < new_right and new_top < new_bottom:
            return BoundingBox(
                left=new_left,
                top=new_top,
                right=new_right,
                bottom=new_bottom,
                width=new_right - new_left,
                height=new_bottom - new_top
            )
        return None

@dataclass
class Center:
    """Represents the center coordinates of a UI element."""
    x: int
    y: int

    def to_string(self) -> str:
        return f'({self.x},{self.y})'

@dataclass
class TreeElementNode:
    """Represents an interactive UI element in the accessibility tree."""
    bounding_box: BoundingBox
    center: Center
    role: str = ''
    subrole: str = ''
    name: str = ''
    description: str = ''
    value: str = ''
    window_name: str = ''
    is_enabled: bool = True
    is_focused: bool = False
    element_type: str = ''
    actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self, index: int) -> list:
        """Convert to a table row for display."""
        label = self.name or self.description or self.value or self.subrole or "(no label)"
        return [index, self.window_name, self.role, label, self.center.to_string(), self.is_focused]

@dataclass
class ScrollElementNode:
    """Represents a scrollable UI element in the accessibility tree."""
    name: str
    role: str
    window_name: str
    bounding_box: BoundingBox
    center: Center
    horizontal_scrollable: bool = False
    horizontal_scroll_percent: float = 0.0
    vertical_scrollable: bool = False
    vertical_scroll_percent: float = 0.0
    is_focused: bool = False
    element_type: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self, index: int, base_index: int) -> list:
        """Convert to a table row for display."""
        return [
            base_index + index,
            self.window_name,
            self.role,
            self.name,
            self.center.to_string(),
            json.dumps(self.metadata)
        ]

@dataclass
class TextElementNode:
    """Represents a text element for DOM/content scraping."""
    text: str

@dataclass
class TreeState:
    """Represents the current state of the accessibility tree."""
    status: bool = True
    root_node: Optional[TreeElementNode] = None
    dom_node: Optional[ScrollElementNode] = None
    interactive_nodes: list[TreeElementNode] = field(default_factory=list)
    scrollable_nodes: list[ScrollElementNode] = field(default_factory=list)
    dom_informative_nodes: list[TextElementNode] = field(default_factory=list)

    def interactive_elements_to_string(self) -> str:
        """Convert interactive elements to a pipe-separated string format."""
        parts = []
        if not self.status:
            parts.append(WARNING_MESSAGE)
            return "\n".join(parts)
        if not self.interactive_nodes and self.status:
            parts.append(EMPTY_MESSAGE)
            return "\n".join(parts)
        # Pipe-separated values with clear header
        header = "# id|window|control_type|name|coords|metadata"
        rows = [header]
        for idx, node in enumerate(self.interactive_nodes):
            metadata = dict(node.metadata)
            metadata['has_focused'] = node.is_focused
            row = f"{idx}|{node.window_name}|{node.element_type or node.role}|{node.name}|{node.center.to_string()}|{json.dumps(metadata)}"
            rows.append(row)
        parts.append("\n".join(rows))
        return "\n".join(parts)

    def scrollable_elements_to_string(self) -> str:
        """Convert scrollable elements to a pipe-separated string format."""
        parts = []
        if not self.status:
            parts.append(WARNING_MESSAGE)
            return "\n".join(parts)
        if not self.scrollable_nodes and self.status:
            parts.append(EMPTY_MESSAGE)
            return "\n".join(parts)
        # Pipe-separated values with clear header
        header = "# id|window|control_type|name|coords|metadata"
        rows = [header]
        base_index = len(self.interactive_nodes)
        for idx, node in enumerate(self.scrollable_nodes):
            metadata = dict(node.metadata)
            metadata['has_focused'] = node.is_focused
            row = (f"{base_index + idx}|{node.window_name}|{node.element_type or node.role}|{node.name}|"
                   f"{node.center.to_string()}|{json.dumps(metadata)}")
            rows.append(row)
        parts.append("\n".join(rows))
        return "\n".join(parts)

# Type alias for any element node type
ElementNode = TreeElementNode | ScrollElementNode | TextElementNode
