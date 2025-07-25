# Relationship Enhancement Plan

## Current State Analysis

The relationships module already includes several key relationship types:
- `Contains` / `IsPartOf` - for embedded objects like doors/windows in walls
- `PassesThrough` / `HasPassingThrough` - for objects passing through others
- `EmbeddedIn` / `Embeds` - for fully embedded objects
- `ConnectsTo` - for physical connections
- `AdjacentTo` - for neighboring objects
- `Supports` / `SupportedBy` - for structural support
- And others for components and creation relationships

## Missing Relationship Types for Building Models

Based on typical building modeling needs, we should add:

### Flow-Based Relationships
- `FlowsTo` / `FlowsFrom` - directional flow relationships for pipes, ducts, cables

### Spatial Relationships  
- `Above` / `Below` - vertical positioning
- `InFrontOf` / `Behind` - depth positioning
- `LeftOf` / `RightOf` - lateral positioning

### Building-Specific Relationships
- `ServesSpace` - for HVAC/electrical serving rooms
- `ServedBySystem` - inverse relationship
- `AccessesSpace` - for doors accessing rooms
- `ProvidesAccessTo` - inverse relationship

### Network/System Relationships
- `PartOfSystem` - for components in building systems
- `HasSystemComponent` - inverse relationship
- `DistributesTo` - for distribution networks
- `ReceivesFrom` - inverse relationship

## Implementation Plan

1. ✅ Add missing relationship classes to relationships.py
2. Update any relationship handling logic to support new types  
3. Add tests for new relationship types
4. Update documentation with examples of usage

## Implementation Status

### ✅ Completed
- Added FlowsTo/FlowsFrom directional flow relationships
- Added spatial positioning relationships (Above/Below, InFrontOf/Behind, LeftOf/RightOf)
- Added building-specific relationships (ServesSpace/ServedBySystem, AccessesSpace/ProvidesAccessTo)
- Added system relationships (PartOfSystem/HasSystemComponent, DistributesTo/ReceivesFrom)
- Verified EmbeddedIn relationship works correctly with doors in walls

### 🔧 Remaining
- Add helper methods in items.py for new relationship types (optional)
- Add comprehensive tests for new relationships
- Update examples to demonstrate new relationship types

## Usage Examples

### Flow Relationships
- Pipe flows to fixture: `pipe.relationships.append(FlowsTo(source=pipe, target=fixture))`
- Water flows from main to branch: `branch.relationships.append(FlowsFrom(source=branch, target=main))`

### Spatial Relationships  
- Floor above basement: `floor.relationships.append(Above(source=floor, target=basement))`
- Window in front of wall: `window.relationships.append(InFrontOf(source=window, target=wall))`

### Building System Relationships
- HVAC serves room: `hvac_unit.relationships.append(ServesSpace(source=hvac_unit, target=room))`
- Door accesses space: `door.relationships.append(AccessesSpace(source=door, target=room))`
- Pipe part of plumbing system: `pipe.relationships.append(PartOfSystem(source=pipe, target=plumbing_system))`

### Embedded Relationships (existing)
- Door embedded in wall: `door.add_embedded_in_relationship(wall)`  # This creates both EmbeddedIn and Embeds relationships