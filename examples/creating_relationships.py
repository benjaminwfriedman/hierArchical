from hierarchical.utils import generate_id, plot_items, print_bill_of_materials, print_parts_report
from hierarchical.geometry import Geometry
from hierarchical.items import Element, Component, Wall, Deck, Window, Door
from hierarchical.relationships import Contains, IsPartOf
from hierarchical.units import UnitSystem
import numpy as np

if __name__ == "__main__":
    # --- Dimensions ---
    stud_height = 2.4
    stud_width = 0.038
    stud_depth = 0.089

    plywood_height = 2.4
    plywood_depth = 0.012
    plywood_width_long = 5.0
    plywood_width_short = plywood_width_long - 2 * (2 * plywood_depth + stud_depth)
    short_displacement = plywood_width_long - (2 * plywood_depth + stud_depth)

    wall_thickness = 2 * plywood_depth + stud_depth

    deck_width = plywood_width_long
    deck_depth = plywood_width_long
    deck_thickness = 0.025

    

    # --- Stud Elements ---
    stud_base = Element(
        id=generate_id("stud"),
        name="2x4 Stud",
        type="stud",
        geometry=Geometry.from_prism(
            base_points=[(0, 0), (stud_width, 0), (stud_width, stud_depth), (0, stud_depth)],
            height=stud_height
        ).forward(plywood_depth),
        material="Metal"
    )
    stud_1 = stud_base
    stud_2 = stud_1.copy(dx=plywood_width_long - stud_width)
    stud_3 = stud_1.copy()
    stud_4 = stud_3.copy(dx=plywood_width_short - stud_width)

    # --- Plywood Elements ---
    def make_plywood(name, width):
        return Element(
            id=generate_id("plywood"),
            name=name,
            type="plywood sheet",
            geometry=Geometry.from_prism(
                base_points=[(0, 0), (width, 0), (width, plywood_depth), (0, plywood_depth)],
                height=plywood_height
            ),
            material="wood"
        )

    plywood_1 = make_plywood("Plywood Long 1", plywood_width_long)
    plywood_2 = plywood_1.copy(dy=stud_depth + plywood_depth)
    plywood_3 = make_plywood("Plywood Short 1", plywood_width_short)
    plywood_4 = plywood_3.copy(dy=stud_depth + plywood_depth)

    # --- Wall Assemblies ---
    wall_assembly_1_long = Component.from_elements((plywood_1, stud_1), name="Wall Long Assembly 1")
    wall_assembly_2_long = Component.from_elements((plywood_2, stud_2), name="Wall Long Assembly 2")
    wall_assembly_1_short = Component.from_elements((plywood_3, stud_3), name="Wall Short Assembly 1")
    wall_assembly_2_short = Component.from_elements((plywood_4, stud_4), name="Wall Short Assembly 2")

    # --- Walls ---
    wall_long_1 = Wall.from_components((wall_assembly_1_long, wall_assembly_2_long), name="South Wall L")
    wall_long_2 = wall_long_1.copy().forward(plywood_width_short + wall_thickness)
    wall_long_2.name = "North Wall L"

    wall_short_1 = Wall.from_components((wall_assembly_1_short, wall_assembly_2_short), name="West Wall S")
    wall_short_1.rotate_z(np.pi / 2).move(dx=wall_thickness, dy=wall_thickness)

    wall_short_2 = wall_short_1.copy().right(short_displacement + wall_thickness / 2)
    wall_short_2.name = "East Wall S"

    # --- Floor Deck ---
    subfloor_panel = Element(
        id=generate_id("subfloor"),
        name="Subfloor Panel",
        type="subfloor panel",
        geometry=Geometry.from_prism(
            base_points=[(0, 0), (deck_width, 0), (deck_width, deck_depth), (0, deck_depth)],
            height=deck_thickness
        ),
        material="wood"
    )

    subfloor_component = Component.from_elements((subfloor_panel,), name="Subfloor Component")
    deck = Deck.from_components((subfloor_component,), name="Floor Deck")
    deck2 = deck.copy().right(deck_width)

    # --- Extra Walls (duplicated for second deck) ---
    wall_long_3 = wall_long_2.copy().right(plywood_width_long)
    wall_long_4 = wall_long_1.copy().right(plywood_width_long)
    wall_short_3 = wall_short_2.copy().right(plywood_width_long)

    # --- Elevate all walls above deck ---
    for wall in [wall_long_1, wall_long_2, wall_long_3, wall_long_4, wall_short_1, wall_short_2, wall_short_3]:
        wall.up(deck_thickness)



        # Wooden side beam: 4in x 4in x 8ft
    beam = Element(
        id=generate_id('beam'),
        name="wooden beam",
        type="beam",
        material="wood",
        geometry=Geometry.from_prism(
            base_points=[(0, 0), (4 , 0), (4, 4), (0, 4)],
            height=8 * 12  # 8ft in inches
        ),
        unit_system=UnitSystem.INCH
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
        unit_system=UnitSystem.INCH
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
        unit_system=UnitSystem.INCH
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
        unit_system=UnitSystem.INCH
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
        unit_system=UnitSystem.INCH
    )

    door_component = Component.from_elements(
        name="door_component",
        type="door_component",
        elements=[sheet],
        unit_system=UnitSystem.INCH
    )

    # Create full Door Object
    created_door = Door.from_components(
        name="door",
        swing_direction="left",
        components=[frame, door_component],
        unit_system=UnitSystem.INCH
    )

    created_door.convert_to_metric()

    created_door.up(deck_thickness)
    # Position door to create substantial overlap with the south wall (wall_long_1)
    # Move door into the wall volume to create actual overlap
    created_door.move(dx=1.0, dy=0.02)  # Move slightly into the wall to create volumetric overlap

    # --- Plot and Report ---
    objects = [wall_long_1, wall_long_2, wall_short_1, wall_short_2,
               deck, deck2, wall_long_3, wall_long_4, wall_short_3]
    
    # Add door to objects list - embedded relationships will be inferred if using Model.from_objects
    objects.append(created_door)
    
    # Embedded relationships will now be inferred automatically by Model.from_objects()
    # with 95% overlap threshold and only for embeddable objects (doors, windows)

    for object in objects:
        adjacent_items = object.find_adjacent_items(objects, tolerance=0.01)
        for adjacent_item in adjacent_items:
            object.add_adjacent_to_relationship(adjacent_item)
    
    # plot_items(objects, color_by_class=True)

 
    print("All Relationships")
    for object in objects:
        for r in object.relationships:
            print(f"{r.source.name} ---{r.type}-> {r.target.name}")
        


    # Querying Relationships
    print("")
    print("There is a structural issue in wall_long_1")

    print("")
    print("Impacted items:")
    impacted_objects = []
    for r in wall_long_1.relationships:
        
        print(f"{r.target.name} is impacted because of {r.type} relationship")
        print(f"{r.target.name}'s componants are impacted as well:")
        impacted_objects.append(r.target)
        for c in r.target.sub_items:
            print(f"    -- {c.name}")

    

    
    print_parts_report(impacted_objects)

    print([o.name for o in impacted_objects])
    