# hierArchical

**hierArchical** is a lightweight, expressive data modeling framework for architecture, built from the ground up around hierarchy, geometry, and computation. Instead of modeling buildings as a pile of files, *hierArchical* treats them as connected systems of parts—ready for reasoning, automation, and simulation.

---

## 🔍 Why hierArchical?

The AEC industry is trapped in document-based tools:  
- **Revit** locks geometry inside opaque, file-based databases.  
- **IFC** is too rigid and verbose for dynamic workflows.  
- **PDFs, DWGs, and Excel** are static snapshots—not systems.

This limits what we can do with architectural data. AI, automation, and even basic querying struggle to operate when information isn't structured or computable.

---

## 🧱 What hierArchical Provides

**hierArchical** introduces a clean data structure and API that treats architecture as a nested, connected, and geometric system:

### ✅ Hierarchy  
From studs to walls to zones to spaces, everything is built from smaller parts.

### ✅ Geometry  
Each item—whether an element or an assembly—has real, computable 3D geometry (B-rep and mesh).

### ✅ Composition  
Objects are made of components, which are made of elements. Geometry and materials flow upward. Recursive access is easy.

### ✅ Transformations  
Objects can be moved, rotated, duplicated, and queried like real-world things.

### ✅ Visualization  
Render your system in 3D with clear colors, class grouping, and coordinate annotations via Plotly.

---

# 🔧 Core Concepts

| Class           | Description                             |
| --------------- | --------------------------------------- |
| `Element`       | The atomic building block (e.g., stud)  |
| `Component`     | Assembly of elements (e.g., wall panel) |
| `Object`        | Functional unit (e.g., wall, floor)     |
| `Wall` / `Deck` | Specializations of `Object`             |


Each class supports:

* `.copy(), .move(), .rotate_z()`

* Recursive access to child parts

* Material aggregation by volume and percent

---

# 📈 Vision
This is a foundation for a new kind of architectural platform—one where data is:

* Composable

* Queryable

* Simulation-Ready

* AI-Assistive

The goal isn’t to replace design tools. It’s to structure design data so it can be used intelligently—by humans, systems, and models alike.

---

# 📍 Status
hierArchical is early-stage but already includes:

* B-rep + mesh geometry system

* Transformable primitives (box, prism, etc.)

* Full element-component-object hierarchy

* Plotly visualization

* Material tracking + reporting

More is coming: zones, spans, structural logic, simulation hooks, and generative tools.