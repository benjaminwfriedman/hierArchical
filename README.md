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
