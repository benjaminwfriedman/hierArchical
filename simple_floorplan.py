# from utils import generate_id, plot_items, print_bill_of_materials, print_parts_report
# from geometry import Geometry
# from items import Element, Component, Wall, Deck
# import numpy as np


# if __name__ == "__main__":
#     stud_height = 2.4
#     stud_width = 0.038
#     stud_depth = 0.089

#     plywood_height = 2.4
#     plywood_depth = 0.012
#     plywood_width_long = 5
#     plywood_width_short = plywood_width_long - 2 * (2 * plywood_depth + stud_depth)  # Adjusted to fit between
#     short_displacement = plywood_width_long - (2 * plywood_depth + stud_depth)

#     wall_thickness = 2 * 0.012 + stud_depth

#     deck_width = plywood_width_long
#     deck_depth = plywood_width_long
#     deck_thickness = 0.025

#     # Base stud
#     stud_1 = Element(
#         id=generate_id("stud"),
#         name="2x4 Stud 1",
#         type="stud",
#         geometry=Geometry.from_prism(
#             base_points=[(0, 0), (stud_width, 0), (stud_width, stud_depth), (0, stud_depth)],
#             height=stud_height
#         ).forward(plywood_depth),
#         material="Metal"
#     )
#     stud_2 = stud_1.copy(dx=plywood_width_long - stud_width)

#     stud_3 = stud_1.copy()
#     stud_4 = stud_3.copy(dx=plywood_width_short - stud_width)

#     # Long plywood
#     plywood_1 = Element(
#         id=generate_id("plywood"),
#         name="Plywood Sheet Long 1",
#         type="plywood sheet",
#         geometry=Geometry.from_prism(
#             base_points=[(0, 0), (plywood_width_long, 0), (plywood_width_long, plywood_depth), (0, plywood_depth)],
#             height=plywood_height
#         ),
#         material="wood"
#     )
#     plywood_2 = plywood_1.copy(dy=stud_depth + plywood_depth)

#     # Short plywood
#     plywood_3 = Element(
#         id=generate_id("plywood"),
#         name="Plywood Sheet Short 1",
#         type="plywood sheet",
#         geometry=Geometry.from_prism(
#             base_points=[(0, 0), (plywood_width_short, 0), (plywood_width_short, plywood_depth), (0, plywood_depth)],
#             height=plywood_height
#         ),
#         material="wood"
#     )
#     plywood_4 = plywood_3.copy(dy=stud_depth + plywood_depth)

#     # Component assemblies
#     wall_assembly_1_long = Component.from_elements((plywood_1, stud_1), name="Wall Long Assembly 1")
#     wall_assembly_2_long = Component.from_elements((plywood_2, stud_2), name="Wall Long Assembly 2")

#     wall_assembly_1_short = Component.from_elements((plywood_3, stud_3), name="Wall Short Assembly 1")
#     wall_assembly_2_short = Component.from_elements((plywood_4, stud_4), name="Wall Short Assembly 2")

#     # Walls
#     wall_long_1 = Wall.from_components((wall_assembly_1_long, wall_assembly_2_long), name="South Wall L")
#     wall_long_2 = wall_long_1.copy()
#     wall_long_2.name = "North Wall L"

#     wall_short_1 = Wall.from_components((wall_assembly_1_short, wall_assembly_2_short), name="West Wall S")
#     wall_short_1.rotate_z(np.pi / 2).move(dx=0, dy=plywood_depth)
#     wall_short_1.forward(wall_thickness).right(wall_thickness).back(plywood_depth)

#     wall_short_2 = wall_short_1.copy()
#     wall_short_2 = wall_short_2.right(short_displacement)


#     wall_long_2 = wall_long_2.forward(plywood_width_short + wall_thickness)



#     # Floor deck
#     subfloor_panel = Element(
#         id=generate_id("subfloor"),
#         name="Subfloor Panel",
#         type="subfloor panel",
#         geometry=Geometry.from_prism(
#             base_points=[(0, 0), (deck_width, 0), (deck_width, deck_depth), (0, deck_depth)],
#             height=deck_thickness
#         ),
#         material="wood"
#     )

#     deck = Deck.from_components(
#         (Component.from_elements((subfloor_panel,), name="Subfloor Component"),),  # <-- comma here
#         name="Floor Deck"
#     )

#     deck2 = deck.copy()
#     deck2 = deck2.right(deck_width)

#     wall_long_3 = wall_long_2.copy()
#     wall_long_3 = wall_long_3.right(plywood_width_long)
    
#     wall_long_4 = wall_long_1.copy()
#     wall_long_4 = wall_long_4.right(plywood_width_long)

#     wall_short_3 = wall_short_2.copy()
#     wall_short_3 = wall_short_3.right(plywood_width_long)

#     # Elevate all walls to sit on top of deck
#     wall_long_1.up(deck_thickness)
#     wall_long_2.up(deck_thickness)
#     wall_short_1.up(deck_thickness)
#     wall_short_2.up(deck_thickness)
#     wall_long_3.up(deck_thickness)
#     wall_long_4.up(deck_thickness)
#     wall_short_3.up(deck_thickness)

#     wall_short_2 = wall_short_2.right(wall_thickness / 2)

#     # # Plot
#     objects = [wall_long_1, wall_long_2, wall_short_1, wall_short_2, deck, deck2, wall_long_3, wall_long_4, wall_short_3]

#     plot_items(objects, color_by_class=True, flatten_to_elements=False)
#     print_bill_of_materials(objects)
#     print_parts_report(objects)


from utils import generate_id, plot_items, print_bill_of_materials, print_parts_report
from geometry import Geometry
from items import Element, Component, Wall, Deck
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

    # --- Plot and Report ---
    objects = [wall_long_1, wall_long_2, wall_short_1, wall_short_2,
               deck, deck2, wall_long_3, wall_long_4, wall_short_3]

    plot_items(objects, color_by_class=True, flatten_to_elements=False)
    print_bill_of_materials(objects)
    print_parts_report(objects)

