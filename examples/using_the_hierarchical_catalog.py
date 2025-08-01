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
building_width = 50.0  # feet
building_depth = 25.0  # feet
wall_height = 10.0      # feet
wall_thickness = 0.5   # feet (6 inches)

walls = []

# in this case the south wall will be the first wall created and will be at 0,0,0 the south west corner

south_wall = ExteriorWall(
    name="South Wall",
    length=building_width,
    height=wall_height,
    thickness=wall_thickness
)

# next we will add the west wall which will be next closest to 0,0,0 and rotated 90 degrees
west_wall = ExteriorWall(
    name="West Wall",
    length=building_depth,
    height=wall_height,
    )

# now we will rotate the west wall 90 degrees to the left (counter-clockwise)
west_wall.rotate_z(math.pi / 2)

# now we will move it up the y axis by the thickness of the south wall and up the x axis by the thickness of the west wall
west_wall.move(dy=south_wall.attributes.width, dx=west_wall.attributes.width)


# now we will add the north wall which will be next closest to 0,0,0 and rotated 180
north_wall = ExteriorWall(
    name="North Wall",
    length=building_width,
    height=wall_height)

north_wall.rotate_z(math.pi)
# now we will move it up the y axis by the thickness of the south wall and up the x axis by the length of the west wall
north_wall.move(dy=west_wall.attributes.length + north_wall.attributes.width, dx=north_wall.attributes.length)


# now we will add the east wall which will be next closest to 0,0,0 and rotated 90 degrees
east_wall = ExteriorWall(
    name="East Wall",
    length=building_depth,
    height=wall_height,
)
# now we will rotate the east wall 90 degrees to the left (clockwise)
east_wall.rotate_z(math.pi / 2)

# now we will move it up the x axis by the length of the south wall and up the y axis by the thickness of the east wall
east_wall.move(dx=south_wall.attributes.length, dy=north_wall.attributes.width)


# now we will do the interior walls

# the first will be the east west which will be at 0, half the depth, 0
east_west_wall = InteriorWall(
    name="East-West Interior Wall",
    length=building_width - west_wall.attributes.width - east_wall.attributes.width,
    height=wall_height,
    thickness=wall_thickness
)
east_west_wall.move(dy=building_depth / 2 - 0.5 * east_west_wall.attributes.width, dx=west_wall.attributes.width)


# next we will add the north south wall northern section
north_south_wall_north = InteriorWall(
    name="North-South Interior Wall North",
    length=building_depth / 2,
    height=wall_height,
    thickness=wall_thickness
)

north_south_wall_north.rotate_z(math.pi / 2)
north_south_wall_north.move(dx=north_south_wall_north.attributes.width + building_width / 2 - east_west_wall.attributes.width / 2 , dy=building_depth / 2 + 0.5 * east_west_wall.attributes.width)

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
                            dy=north_wall.attributes.width)



plot_items([south_wall, west_wall, north_wall, east_wall, east_west_wall, north_south_wall_north, north_south_wall_south], show_coords=False)

from hierarchical.utils import print_parts_report
print_parts_report([south_wall, west_wall, north_wall, east_wall, east_west_wall, north_south_wall_north, north_south_wall_south])