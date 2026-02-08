"""
Data classes for representing macOS accessibility tree elements and state.
"""
from dataclasses import dataclass, field
from tabulate import tabulate
from typing import Optional

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

    def to_row(self, index: int, base_index: int) -> list:
        """Convert to a table row for display."""
        return [
            base_index + index,
            self.window_name,
            self.role,
            self.name,
            self.center.to_string(),
            self.horizontal_scrollable,
            self.horizontal_scroll_percent,
            self.vertical_scrollable,
            self.vertical_scroll_percent,
            self.is_focused
        ]

@dataclass
class TextElementNode:
    """Represents a text element for DOM/content scraping."""
    text: str

@dataclass
class TreeState:
    """Represents the current state of the accessibility tree."""
    root_node: Optional[TreeElementNode] = None
    dom_node: Optional[ScrollElementNode] = None
    interactive_nodes: list[TreeElementNode] = field(default_factory=list)
    scrollable_nodes: list[ScrollElementNode] = field(default_factory=list)
    dom_informative_nodes: list[TextElementNode] = field(default_factory=list)

    def interactive_elements_to_string(self) -> str:
        """Convert interactive elements to a pipe-separated string format."""
        if not self.interactive_nodes:
            return "No interactive elements"
        header = "# id|window|role|type|name|coords|focus"
        rows = [header]
        for idx, node in enumerate(self.interactive_nodes):
            label = node.name or node.description or node.value or "(no label)"
            row = f"{idx}|{node.window_name}|{node.role}|{node.element_type}|{label}|{node.center.to_string()}|{node.is_focused}"
            rows.append(row)
        return "\n".join(rows)

    def scrollable_elements_to_string(self) -> str:
        """Convert scrollable elements to a pipe-separated string format."""
        if not self.scrollable_nodes:
            return "No scrollable elements"
        header = "# id|window|role|name|coords|h_scroll|h_pct|v_scroll|v_pct|focus"
        rows = [header]
        base_index = len(self.interactive_nodes)
        for idx, node in enumerate(self.scrollable_nodes):
            row = (f"{base_index + idx}|{node.window_name}|{node.role}|{node.name}|"
                   f"{node.center.to_string()}|{node.horizontal_scrollable}|{node.horizontal_scroll_percent}|"
                   f"{node.vertical_scrollable}|{node.vertical_scroll_percent}|{node.is_focused}")
            rows.append(row)
        return "\n".join(rows)

# Type alias for any element node type
ElementNode = TreeElementNode | ScrollElementNode | TextElementNode
