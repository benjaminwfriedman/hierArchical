# Dataset

## An Ideal

AEC is severely hampered by the data structures we've been forced into. These include:

- **Revit**  
  A powerful modeling tool, but fundamentally built around proprietary, file-based data that resists integration, querying, and real-time interaction.

- **IFC (Industry Foundation Classes)**  
  Intended as a neutral standard, but too rigid and verbose for dynamic workflows, and poorly supported by many tools in practice.

- **PDFs and DWGs**  
  Ubiquitous in documentation but functionally dead ends from a data perspective. They capture snapshots, not systems.

- **Excel**  
  Still the de facto database for many firms, but siloed, versioned, and disconnected from spatial or model logic.

- **Folder-Based Storage**  
  Whether on local drives or cloud file systems, this reinforces isolation between projects, disciplines, and tools. No shared ontology, no discoverability.

---

### The Result

Data is locked in formats that were optimized for **files**, not **systems**.  
They were designed for **drawings**, not **decisions**.

Until we build infrastructure that treats AEC data as **queryable**, **connected**, and **computable**, AI and automation will remain limited to surface-level tasks.

---

## A Solution

To solve this conundrum, we must start with first principles of architecture and build up a descriptive architectural dataset from the ground up.

---

## Requirements

A quality dataset for architecture **must** be:

- **Hierarchical**  
  Able to represent the nested structure of buildings—from campus to room to detail.

- **Relational**  
  Capturing how elements connect and depend on each other—across disciplines, phases, and systems.

- **Visual**  
  Supporting both geometry and the human-readable representation of spatial ideas.

- **Shapable**  
  Structured in a way that supports geometric manipulation, pattern recognition, and rule-based design systems (like shape grammars). The data should enable iterative exploration and generative workflows.

- **Standardized**  
  So data can be reused across tools, projects, and firms without loss of fidelity or meaning.

- **Obey the Laws of Physics**  
  There is a strange tendency in AEC software for things to “sort of” be somewhere—or to show up twice so they can be seen in two different views. Every object should be represented as it is: clearly and unambiguously.

---

## An Idea

I believe that architecture is natural—an emergent whole composed of interrelated parts. To represent it faithfully, we must start from the bottom up.

---

## The Hierarchy

---

### Elements

The smallest units of architecture—the literal building blocks.  
**Examples:** 2x4 lumber, screws, glue, metal sheets

---

### Components

Assemblies made from elements that form recognizable architectural sub-elements.  
**Examples:** moulding, a door lock, a cabinet drawer

---

### Objects

Self-contained units composed of components that form the physical fabric of a building. Objects are typically split by room or enclosure boundaries.  
**Examples:** ducts, walls, doors, furniture, windows, decks (floor on the above level, ceiling on the below level), built-in casework, beams

> Objects can span or belong to multiple zones, spaces, levels, or even both interior and exterior classifications.

---

### Zones

Functional groupings of objects that work together to support a particular activity or transition.  
**Examples:**  
- Entry zone (door, coat rack)  
- Circulation zone (rug, display table)  
- Lounge zone (couch, coffee table, TV)

---

### Spaces

Coherent assemblies of zones that deliver a complete experience or function. While often aligned with rooms, they need not be fully enclosed.  
**Examples:** living space, sleeping space, caring space, recovery space, conference space

---

### Bounds

Bounds define the transitions between spaces—whether fully enclosed, semi-enclosed, or open.  
**Examples:**  
- External closed bound (wall)  
- External door bound (wall + door)  
- External window bound (wall + window)  
- Internal closed bound (interior wall)  
- Internal door/window bound  
- Internal open bound (intentional spatial opening)

---

### Spans

Tissues of continuity—elements that stretch across space boundaries when they are truly shared.  
**Example:** a floor span between two connected spaces

---

### Levels

Horizontally contiguous sets of spaces.  
**Examples:** ground level, second floor

---

### Structural Systems

Structural systems are higher-order assemblies that organize multiple physical components into a coherent load-bearing or force-resisting whole. They define how a building stands up, resists environmental forces, and distributes loads safely.  
**Examples:** gravity systems, lateral systems, diaphragm systems, foundational systems

---

### Building

A complete architectural entity composed of levels, spaces, bounds, objects, and all sub-elements described above.

This hierarchy provides a foundation for a **queryable**, **spatially-aware**, and **semantically rich** architectural dataset—one capable of supporting automation, simulation, and AI-assisted design.

---

## More on Objects

Objects are one of the most versatile categories in this system. They can be thought of as the smallest item that is an "item" in its own right. For example, a wall is an object made up of components like drywall and studs, which individually are not objects—but together form one.

---

### Wall

Walls are made up of **components** like:

- Wall assemblies  
- Beams  
- Moulding

#### Shape

Walls are three-dimensional and can be described as:

- A set of 3D components (e.g., studs, drywall)  
- The bounding shell of those components  
- The union (or "shrink wrap") of those components

**B-rep:** precise modeling of edges, faces, and vertices  
**Triangle mesh:** simplified geometry for rendering or simulation

> Walls are either **planar** or **curved**—but not zigzagged or topographic.  
> Walls may bend, but they may not corner.  
> Any sharp directional change requires splitting into multiple wall objects.

#### Relationships

- Can be part of multiple spaces  
- Can be part of a boundary  
- Can contain windows and doors  
- Can support mounted elements  
- Can include elements such as beams, pipes, wires

#### Attributes

- Sound penetration (computed via sub-elements)  
- Embodied GHG  
- Fire rating  
- Surface finish

#### Ontological Classifications

- Internal / External  
- Load-bearing / Non-load-bearing  
- Soundproof / Not soundproof  

#### Graph Representation

- Edge  
- Point

---

### Pipe

Pipes are made up of components such as:

- Straight segments  
- Elbows  
- Tees  
- Couplers  
- Insulation  
- Valves

#### Shape

**B-rep:** shell and joint representation  
**Triangle mesh:** lightweight modeling for rendering or simulation

#### Relationships

- Can be included in: walls, spaces, decks, roofs  
- Can connect to: sinks, shower heads, toilets, water heaters, appliances

#### Attributes

- Embodied GHG  
- Flow capacity  
- Thermal conductivity  
- Pressure rating  
- Insulation type

#### Ontological Classifications

- Internal / External  
- Supply / Return / Drain  
- Potable / Fire protection / Greywater

#### Graph Representation

- Edges (pipe segments)  
- Nodes (connections or endpoints)

---

### Deck

Decks are made up of subcomponents such as:

- Floor assemblies  
- Ceiling assemblies  
- Joists, beams, and trusses  
- Insulation layers  
- Finishes (e.g., hardwood, carpet)  
- Embedded systems (e.g., radiant heating)

#### Shape

- As a set of 3D components  
- As a bounding shell  
- As a union ("shrink wrap") volume

#### Relationships

- Can be included in: levels, structural systems, spaces  
- Can contain: ducts, pipes, ceiling systems, access panels, floor fixtures

#### Attributes

- Fire rating  
- Acoustic rating  
- Thermal resistance (R-value)  
- Span capacity  
- Opening count/area

#### Ontological Classifications

- Structural / Non-structural  
- Raised / Flush  
- Accessible / Not accessible  
- Interior / Exterior  

#### Graph Representation

- Edge  
- Point

---

## Use Cases

This dataset structure—hierarchical, relational, geometric, and semantically rich—unlocks a wide range of use cases across disciplines, workflows, and automation pipelines.

---

### Design Intelligence

- Automated adjacency analysis  
- Rule-based layout generation  
- Furniture and equipment optimization

### Construction + Fabrication

- Precise quantity takeoffs  
- Automated assembly instructions  
- Prefabrication packaging

### Coordination + Clash Detection

- Multi-system collision detection  
- Zone-based filtering  
- Surface-based alignment

### Systems Reasoning + Simulation

- Graph-based flow modeling  
- Load path and deflection analysis  
- Daylight or energy modeling

### Archival + Audit

- Change detection over time  
- Code compliance checking  
- Permitting and documentation automation

### AI & Automation

- Semantic search and retrieval  
- ML training data generation  
- Real-time generative design feedback

---

This system turns architectural data from a set of disconnected files into a **computational design platform**—ready for automation, analysis, and creative augmentation.
