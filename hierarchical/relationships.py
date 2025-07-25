from dataclasses import dataclass, field
from typing import Dict, Any
from .helpers import generate_id
from uuid import uuid4

@dataclass(slots=True)
class Relationship:
    """A relationship between two items, e.g., 'contains', 'is part of'."""
    source: str
    target: str
    type: str
    id: str = field(default_factory=lambda: str(uuid4()))
    attributes: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))

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

@dataclass(slots=True)
class Creates(Relationship):
    """Object 1 creates object 2 (e.g., a tool creating a part)."""
    type: str = field(default="creates", init=False)

@dataclass(slots=True)
class FlowsTo(Relationship):
    """Directional flow from object 1 to object 2 (e.g., pipe flows to fixture)."""
    type: str = field(default="flows_to", init=False)

@dataclass(slots=True)
class FlowsFrom(Relationship):
    """Directional flow from object 2 to object 1 (inverse of FlowsTo)."""
    type: str = field(default="flows_from", init=False)

@dataclass(slots=True)
class Above(Relationship):
    """Object 1 is positioned above object 2."""
    type: str = field(default="above", init=False)

@dataclass(slots=True)
class Below(Relationship):
    """Object 1 is positioned below object 2 (inverse of Above)."""
    type: str = field(default="below", init=False)

@dataclass(slots=True)
class InFrontOf(Relationship):
    """Object 1 is positioned in front of object 2."""
    type: str = field(default="in_front_of", init=False)

@dataclass(slots=True)
class Behind(Relationship):
    """Object 1 is positioned behind object 2 (inverse of InFrontOf)."""
    type: str = field(default="behind", init=False)

@dataclass(slots=True)
class LeftOf(Relationship):
    """Object 1 is positioned to the left of object 2."""
    type: str = field(default="left_of", init=False)

@dataclass(slots=True)
class RightOf(Relationship):
    """Object 1 is positioned to the right of object 2 (inverse of LeftOf)."""
    type: str = field(default="right_of", init=False)

@dataclass(slots=True)
class ServesSpace(Relationship):
    """Object 1 serves space 2 (e.g., HVAC unit serves room)."""
    type: str = field(default="serves_space", init=False)

@dataclass(slots=True)
class ServedBySystem(Relationship):
    """Object 1 is served by system 2 (inverse of ServesSpace)."""
    type: str = field(default="served_by_system", init=False)

@dataclass(slots=True)
class AccessesSpace(Relationship):
    """Object 1 provides access to space 2 (e.g., door accesses room)."""
    type: str = field(default="accesses_space", init=False)

@dataclass(slots=True)
class ProvidesAccessTo(Relationship):
    """Object 1 provides access via object 2 (inverse of AccessesSpace)."""
    type: str = field(default="provides_access_to", init=False)

@dataclass(slots=True)
class PartOfSystem(Relationship):
    """Object 1 is a component of system 2."""
    type: str = field(default="part_of_system", init=False)

@dataclass(slots=True)
class HasSystemComponent(Relationship):
    """System 1 has component 2 (inverse of PartOfSystem)."""
    type: str = field(default="has_system_component", init=False)

@dataclass(slots=True)
class DistributesTo(Relationship):
    """Object 1 distributes to object 2 (e.g., main line distributes to branch)."""
    type: str = field(default="distributes_to", init=False)

@dataclass(slots=True)
class ReceivesFrom(Relationship):
    """Object 1 receives from object 2 (inverse of DistributesTo)."""
    type: str = field(default="receives_from", init=False)