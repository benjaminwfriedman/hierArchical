import ifcopenshell
from heirarchical.items import Object, Door, Element, Component
from heirarchical.utils import plot_items
from heirarchical.geometry import Geometry
from heirarchical.helpers import random_color, generate_id

# ---------------------------
# Load Door from IFC
# ---------------------------
door = Door.from_ifc(
    ifc_path="content/GEALAN_S9000_Front-Door-IFC/GEALAN_S9000_Door_1100x2000-IFC2X3.ifc"
)

# ---------------------------
# Build Door from Elements
# ---------------------------

# Wooden side beam: 4in x 4in x 8ft
beam = Element(
    id=generate_id('beam'),
    name="wooden beam",
    type="beam",
    material="wood",
    geometry=Geometry.from_prism(
        base_points=[(0, 0), (4, 0), (4, 4), (0, 4)],
        height=8 * 12  # 8ft in inches
    ),
)

# Short bottom beam: 3ft x 4in x 2in
short_beam = Element(
    id=generate_id('beam'),
    name="short wooden beam",
    type="beam",
    material="wood",
    geometry=Geometry.from_prism(
        base_points=[(0, 0), (36, 0), (36, 4), (0, 4)],
        height=2
    ),
)
short_beam.right(4)

# Top header beam: 3ft x 4in x 4in, positioned at top
header_beam = Element(
    id=generate_id('header_beam'),
    name="header beam",
    type="beam",
    material="wood",
    geometry=Geometry.from_prism(
        base_points=[(0, 0), (36, 0), (36, 4), (0, 4)],
        height=4
    ),
)
header_beam.right(4)
header_beam.up(8 * 12 - 4)

# Central door sheet: 3ft x 2in x (8ft - 4in)
sheet = Element(
    id=generate_id('sheet'),
    name="sheet",
    type="sheet",
    material="wood",
    geometry=Geometry.from_prism(
        base_points=[(0, 0), (36, 0), (36, 2), (0, 2)],
        height=(8 * 12 - 4)
    ),
)
sheet.right(4)
sheet.up(2)

# Right side beam (copy of left beam, shifted)
beam_2 = beam.copy()
beam_2.name = "beam 2"
beam_2.right(3 * 12 + 4)

# ---------------------------
# Assemble Components
# ---------------------------

frame = Component.from_elements(
    name="frame",
    type="frame",
    elements=[beam, beam_2, short_beam, header_beam],
)

door_component = Component.from_elements(
    name="door_component",
    type="door_component",
    elements=[sheet],
)

# Create full Door Object
created_door = Door.from_components(
    name="door",
    swing_direction="left",
    components=[frame, door_component],
)
created_door.right(12 * 8)

# ---------------------------
# Visualize
# ---------------------------
plot_items([door, created_door], color_by_attribute="swing_direction")

print("Done")
