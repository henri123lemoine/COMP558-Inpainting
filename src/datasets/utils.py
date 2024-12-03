from enum import Enum, auto


class ImageCategory(Enum):
    """Categories for synthetic and real images."""

    STRUCTURE = auto()
    TEXTURE = auto()
    GRADIENT = auto()
    REAL = auto()
    CUSTOM = auto()
