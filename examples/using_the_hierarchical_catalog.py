#!/usr/bin/env python3
"""
Comprehensive example using the hierarchical catalog to create a 4-room model
with doors between all rooms.

This demonstrates:
- Importing from hierarchical.catalog
- Creating walls, doors, and structural elements
- Positioning and orienting building components
- Material aggregation through the hierarchy
- Building a complete architectural model
"""

import sys
import os
import math

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from hierarchical catalog
from hierarchical.catalog.elements.lumber import Lumber2X4, Lumber2X6, Lumber4X4, Lumber6X6
from hierarchical.catalog.elements.steel import SteelW8X10, SteelW10X15
from hierarchical.catalog.components.wall_frames import WallFrame2X4, WallFrame2X6
from hierarchical.catalog.objects.walls import ExteriorWall, InteriorWall
from hierarchical.catalog.objects.decks import WoodFramedDeck
from hierarchical.catalog.objects.doors import SwingDoor, SlidingDoor, PocketDoor
from hierarchical.utils import plot_items

import math

## STANDARDIZATION ##

# All items, components, and objects are built starting at 0,0,0 with the X axis being the longest dimension, 
# y being the second longest, and z being the 3rd (or up when up is important)
# This allows for easy alignment and positioning of items in a 3D space.
# This also standardizes how objects must be moved, rotated and scaled to work together.

# --- Wall Dimensions ---
# Wall dimensions are defined in feet for easy architectural scaling
# For walls X dimention is length on creation
# For walls Y dimention is height on creation
# For walls Z dimention is thickness on creation

# when aranging objects always start with the object that will be at 0,0,0 and needs no rotation
# next will be objects at 0,0,0 that need rotation


    
# Building dimensions
building_width = 20.0  # feet
building_depth = 18.0  # feet
wall_height = 8.0      # feet
wall_thickness = 0.5   # feet (6 inches)

objects = []

# first create the floor
ground_deck = WoodFramedDeck(
    name="Ground Deck",
    deck_width=building_depth,
    deck_length=building_width,
    include_floor_assembly=True,
    include_ceiling_assembly=False,  # No ceiling below ground floor
    floor_assembly_type='tile',
    wood_species='Douglas Fir'
)

# in this case the south wall will be the first wall created and will be at 0,0,0 the south west corner




south_wall = ExteriorWall(
    name="South Wall",
    length=building_width,
    height=wall_height,
    thickness=wall_thickness
)

# move up the z by the thickness of the ground deck
south_wall.move(dz=ground_deck.attributes.height)

# next we will add the west wall which will be next closest to 0,0,0 and rotated 90 degrees
west_wall = ExteriorWall(
    name="West Wall",
    length=building_depth - south_wall.attributes.width,
    height=wall_height,
    )

# now we will rotate the west wall 90 degrees to the left (counter-clockwise)
west_wall.rotate_z(math.pi / 2)

# now we will move it up the y axis by the thickness of the south wall and up the x axis by the thickness of the west wall
west_wall.move(dy=south_wall.attributes.width, dx=west_wall.attributes.width, dz=ground_deck.attributes.height)


# now we will add the north wall which will be next closest to 0,0,0 and rotated 180
north_wall = ExteriorWall(
    name="North Wall",
    length=building_width - west_wall.attributes.width,
    height=wall_height)

north_wall.rotate_z(math.pi)
# now we will move it up the y axis by the thickness of the south wall and up the x axis by the length of the west wall
north_wall.move(dy=west_wall.attributes.length + north_wall.attributes.width, dx=north_wall.attributes.length, dz=ground_deck.attributes.height)


# now we will add the east wall which will be next closest to 0,0,0 and rotated 90 degrees
east_wall = ExteriorWall(
    name="East Wall",
    length=building_depth - north_wall.attributes.width,
    height=wall_height,
)
# now we will rotate the east wall 90 degrees to the left (clockwise)
east_wall.rotate_z(math.pi / 2)

# now we will move it up the x axis by the length of the south wall and up the y axis by the thickness of the east wall
east_wall.move(dx=south_wall.attributes.length, dy=north_wall.attributes.width, dz=ground_deck.attributes.height)


# now we will do the interior walls

# the first will be the east west which will be at 0, half the depth, 0
east_west_wall = InteriorWall(
    name="East-West Interior Wall",
    length=building_width - west_wall.attributes.width - east_wall.attributes.width,
    height=wall_height,
    thickness=wall_thickness
)
east_west_wall.move(dy=building_depth / 2 - 0.5 * east_west_wall.attributes.width, dx=west_wall.attributes.width, dz=ground_deck.attributes.height)


# next we will add the north south wall northern section
north_south_wall_north = InteriorWall(
    name="North-South Interior Wall North",
    length=building_depth / 2 - east_west_wall.attributes.width,
    height=wall_height,
    thickness=wall_thickness
)

north_south_wall_north.rotate_z(math.pi / 2)
north_south_wall_north.move(dx=north_south_wall_north.attributes.width + building_width / 2 - east_west_wall.attributes.width / 2 , dy=building_depth / 2 + 0.5 * east_west_wall.attributes.width, dz=ground_deck.attributes.height)

# now we will add the south section of the north south wall
north_south_wall_south = InteriorWall(
    name="North-South Interior Wall South",
    length=building_depth / 2 - east_west_wall.attributes.width / 2 - north_wall.attributes.width,
    height=wall_height,
    thickness=wall_thickness
)
north_south_wall_south.rotate_z(math.pi / 2)
north_south_wall_south.move(dx=north_south_wall_north.attributes.width + 
                                building_width / 2 - 
                                east_west_wall.attributes.width / 2, 
                            dy=north_wall.attributes.width, 
                            dz=ground_deck.attributes.height)

# add ceiling deck
# Create a ceiling deck for the ground floor
ceiling_deck = WoodFramedDeck(
    name="Ceiling Deck",
    deck_width=building_depth,
    deck_length=building_width,
    include_floor_assembly=False,  # No floor above ground floor
    include_ceiling_assembly=True,  # Add ceiling assembly
    ceiling_assembly_type='drywall',  # Use drywall for ceiling
    wood_species='Douglas Fir'
)

# move the ceiling deck up to the top of the walls
ceiling_deck.move(dz=wall_height + ground_deck.attributes.height)   

objects.extend([
    ground_deck,
    south_wall,
    west_wall,
    north_wall,
    east_wall,
    east_west_wall,
    north_south_wall_north,
    north_south_wall_south,
    ceiling_deck
])

# plot_items(objects, show_coords=False)

# build model
if __name__ == '__main__':
    from hierarchical.abstractions import Model
    from hierarchical.utils import print_parts_report
    
    print_parts_report(objects)
    
    model = Model.from_objects("4 Room Building Model", objects)
        

    print("Q: What spaces are in the model?")
    print(model.ask("What spaces are in the model?"))

    model.show()
    model.show_building_graph()
    model.show_spaces()
    model.show_spaces_graph()
