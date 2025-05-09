# 🏗️ Simple Floor Plan Prototype

This repository provides a working prototype that builds a simple floorplan using a structured architectural data model. It demonstrates how architecture can be represented as a **computable**, **queryable**, and **hierarchical** system—unlike traditional file-based formats.

---

## 📦 What This Script Does

The script constructs a basic room using three layers of abstraction:

- **Elements** – individual building blocks (e.g., studs, plywood)
- **Components** – small assemblies (e.g., wall segments)
- **Objects** – complete architectural units (e.g., walls, decks)

Each item includes:

- B-rep geometry (surfaces, edges, and vertices)
- Transformations (translation, rotation)
- Material tracking (volume, percent composition)
- Full object/component/element hierarchy

### Included Features

- 3D visualization of the structure using Plotly
- Bill of materials (by material type)
- Parts report (e.g., "8 x stud", "4 x plywood sheet")
- Recursive traversal of the object hierarchy

---

## 🚀 Getting Started

### Requirements

Ensure you have the following modules in your project:

- `geometry.py` – handles geometry creation and transformation
- `items.py` – defines Elements, Components, and Objects
- `utils.py` – contains helper functions like `generate_id`, visualization, and reports

### Run the Script

```bash
python simple_room.py


