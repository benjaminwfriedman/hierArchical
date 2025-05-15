from dataclasses import dataclass, field
from typing import Dict, Any
from .helpers import generate_id

@dataclass(slots=True)
class Relationship:
    """A relationship between two items, e.g., 'contains', 'is part of'."""
    source: str
    target: str
    type: str
    id: str = field(default="")
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = generate_id(self.type)

    def __repr__(self):
        """Lightweight representation for debugger"""
        return f"{self.__class__.__name__}(id={self.id}, name='{self.name}', type='{self.type}')"
    
    def __str__(self):
        return self.__repr__()

# Use field defaults instead of __post_init__
@dataclass(slots=True)
class Contains(Relationship):
    """A relationship indicating that one item fully contains another."""
    type: str = field(default="contains", init=False)

@dataclass(slots=True)
class IsPartOf(Relationship):
    """A relationship indicating that one item is part of another."""
    type: str = field(default="is_part_of", init=False)

@dataclass(slots=True)
class PassesThrough(Relationship):
    """Object 1 passes through object 2 (e.g., pipe through multiple walls)."""
    type: str = field(default="passes_through", init=False)

@dataclass(slots=True)
class HasPassingThrough(Relationship):
    """Object 1 has object 2 passing through it (inverse of PassesThrough)."""
    type: str = field(default="has_passing_through", init=False)

@dataclass(slots=True)
class EmbeddedIn(Relationship):
    """Object 1 is fully embedded within object 2 (e.g., window in wall)."""
    type: str = field(default="embedded_in", init=False)

@dataclass(slots=True)
class Embeds(Relationship):
    """Object 1 embeds object 2 (inverse of EmbeddedIn)."""
    type: str = field(default="embeds", init=False)

@dataclass(slots=True)
class HasComponent(Relationship):
    """Object has a component as a sub-element."""
    type: str = field(default="has_component", init=False)

@dataclass(slots=True)
class IsComponentOf(Relationship):
    """Object is a component of another (inverse of HasComponent)."""
    type: str = field(default="is_component_of", init=False)

@dataclass(slots=True)
class ConnectsTo(Relationship):
    """Physical connection between objects (e.g., pipe to fixture)."""
    type: str = field(default="connects_to", init=False)

@dataclass(slots=True)
class AdjacentTo(Relationship):
    """Objects that touch or are next to each other."""
    type: str = field(default="adjacent_to", init=False)

@dataclass(slots=True)
class Supports(Relationship):
    """Object 1 structurally supports object 2."""
    type: str = field(default="supports", init=False)

@dataclass(slots=True)
class SupportedBy(Relationship):
    """Object 1 is supported by object 2 (inverse of Supports)."""
    type: str = field(default="supported_by", init=False)