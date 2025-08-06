"""
Parametric steel elements with AISC standard dimensions.
"""

import math
from typing import Dict, List, Tuple
from ..base import ParametricElement, Parameter
from hierarchical.geometry import Geometry


def _create_i_beam_profile(depth: float, flange_width: float, web_thickness: float, 
                          flange_thickness: float) -> List[Tuple[float, float]]:
    """
    Create I-beam profile points for prism geometry.
    
    Args:
        depth: Total depth of beam (inches)
        flange_width: Width of flanges (inches)
        web_thickness: Thickness of web (inches)
        flange_thickness: Thickness of flanges (inches)
    
    Returns:
        List of (x, y) points defining the I-beam cross-section
    """
    # Convert to feet
    d = depth / 12
    bf = flange_width / 12
    tw = web_thickness / 12
    tf = flange_thickness / 12
    
    # Calculate key dimensions
    half_flange = bf / 2
    half_web = tw / 2
    web_height = d - 2 * tf
    
    # Create I-beam profile points (clockwise from bottom-left)
    points = [
        # Bottom flange - left to right
        (-half_flange, 0),                    # Bottom left corner
        (half_flange, 0),                     # Bottom right corner
        (half_flange, tf),                    # Bottom right flange top
        (half_web, tf),                       # Web connection right
        
        # Web - bottom to top
        (half_web, d - tf),                   # Web top right
        (half_flange, d - tf),                # Top flange bottom right
        
        # Top flange - right to left
        (half_flange, d),                     # Top right corner
        (-half_flange, d),                    # Top left corner
        (-half_flange, d - tf),               # Top left flange bottom
        (-half_web, d - tf),                  # Web connection left
        
        # Web - top to bottom
        (-half_web, tf),                      # Web bottom left
        (-half_flange, tf),                   # Bottom flange top left
    ]
    
    return points


class SteelW8X10(ParametricElement):
    """AISC W8x10 Wide Flange Beam"""
    
    # AISC W8x10 properties (inches)
    DEPTH = 7.89
    FLANGE_WIDTH = 3.94
    WEB_THICKNESS = 0.170
    FLANGE_THICKNESS = 0.205
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'length': Parameter(
                name='length',
                type=float,
                default=10.0,
                min_value=1.0,
                max_value=40.0,
                unit='ft',
                description="Length of the steel beam"
            ),
            'grade': Parameter(
                name='grade',
                type=str,
                default='A992',
                description="Steel grade (A992, A572, A36, etc.)"
            )
        }
    
    @classmethod
    def get_material_type(cls) -> str:
        return "steel"
    
    def create_geometry(self) -> Geometry:
        profile_points = _create_i_beam_profile(
            self.DEPTH, self.FLANGE_WIDTH, 
            self.WEB_THICKNESS, self.FLANGE_THICKNESS
        )
        return Geometry.from_prism(profile_points, self.params['length'])


class SteelW8X13(ParametricElement):
    """AISC W8x13 Wide Flange Beam"""
    
    # AISC W8x13 properties (inches)
    DEPTH = 7.99
    FLANGE_WIDTH = 4.00
    WEB_THICKNESS = 0.230
    FLANGE_THICKNESS = 0.255
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'length': Parameter(
                name='length',
                type=float,
                default=10.0,
                min_value=1.0,
                max_value=40.0,
                unit='ft',
                description="Length of the steel beam"
            ),
            'grade': Parameter(
                name='grade',
                type=str,
                default='A992',
                description="Steel grade (A992, A572, A36, etc.)"
            )
        }
    
    @classmethod
    def get_material_type(cls) -> str:
        return "steel"
    
    def create_geometry(self) -> Geometry:
        profile_points = _create_i_beam_profile(
            self.DEPTH, self.FLANGE_WIDTH, 
            self.WEB_THICKNESS, self.FLANGE_THICKNESS
        )
        return Geometry.from_prism(profile_points, self.params['length'])


class SteelW10X15(ParametricElement):
    """AISC W10x15 Wide Flange Beam"""
    
    # AISC W10x15 properties (inches)
    DEPTH = 9.99
    FLANGE_WIDTH = 4.00
    WEB_THICKNESS = 0.230
    FLANGE_THICKNESS = 0.270
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'length': Parameter(
                name='length',
                type=float,
                default=12.0,
                min_value=1.0,
                max_value=40.0,
                unit='ft',
                description="Length of the steel beam"
            ),
            'grade': Parameter(
                name='grade',
                type=str,
                default='A992',
                description="Steel grade (A992, A572, A36, etc.)"
            )
        }
    
    @classmethod
    def get_material_type(cls) -> str:
        return "steel"
    
    def create_geometry(self) -> Geometry:
        profile_points = _create_i_beam_profile(
            self.DEPTH, self.FLANGE_WIDTH, 
            self.WEB_THICKNESS, self.FLANGE_THICKNESS
        )
        return Geometry.from_prism(profile_points, self.params['length'])


class SteelW12X19(ParametricElement):
    """AISC W12x19 Wide Flange Beam"""
    
    # AISC W12x19 properties (inches)
    DEPTH = 12.16
    FLANGE_WIDTH = 4.01
    WEB_THICKNESS = 0.235
    FLANGE_THICKNESS = 0.350
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'length': Parameter(
                name='length',
                type=float,
                default=12.0,
                min_value=1.0,
                max_value=40.0,
                unit='ft',
                description="Length of the steel beam"
            ),
            'grade': Parameter(
                name='grade',
                type=str,
                default='A992',
                description="Steel grade (A992, A572, A36, etc.)"
            )
        }
    
    @classmethod
    def get_material_type(cls) -> str:
        return "steel"
    
    def create_geometry(self) -> Geometry:
        profile_points = _create_i_beam_profile(
            self.DEPTH, self.FLANGE_WIDTH, 
            self.WEB_THICKNESS, self.FLANGE_THICKNESS
        )
        return Geometry.from_prism(profile_points, self.params['length'])


class SteelW16X26(ParametricElement):
    """AISC W16x26 Wide Flange Beam"""
    
    # AISC W16x26 properties (inches)
    DEPTH = 15.69
    FLANGE_WIDTH = 5.50
    WEB_THICKNESS = 0.250
    FLANGE_THICKNESS = 0.345
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'length': Parameter(
                name='length',
                type=float,
                default=16.0,
                min_value=1.0,
                max_value=40.0,
                unit='ft',
                description="Length of the steel beam"
            ),
            'grade': Parameter(
                name='grade',
                type=str,
                default='A992',
                description="Steel grade (A992, A572, A36, etc.)"
            )
        }
    
    @classmethod
    def get_material_type(cls) -> str:
        return "steel"
    
    def create_geometry(self) -> Geometry:
        profile_points = _create_i_beam_profile(
            self.DEPTH, self.FLANGE_WIDTH, 
            self.WEB_THICKNESS, self.FLANGE_THICKNESS
        )
        return Geometry.from_prism(profile_points, self.params['length'])


class SteelTube4X4X1_4(ParametricElement):
    """Steel square tube 4x4x1/4"""
    
    # Tube properties (inches)
    WIDTH = 4.0
    HEIGHT = 4.0
    WALL_THICKNESS = 0.25
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'length': Parameter(
                name='length',
                type=float,
                default=12.0,
                min_value=1.0,
                max_value=40.0,
                unit='ft',
                description="Length of the steel tube"
            ),
            'grade': Parameter(
                name='grade',
                type=str,
                default='A500',
                description="Steel grade (A500, A36, etc.)"
            )
        }
    
    @classmethod
    def get_material_type(cls) -> str:
        return "steel"
    
    def create_geometry(self) -> Geometry:
        # Convert to feet
        outer_w = self.WIDTH / 12
        outer_h = self.HEIGHT / 12
        wall_t = self.WALL_THICKNESS / 12
        
        # Create hollow square profile
        # Outer square (clockwise)
        outer_points = [
            (-outer_w/2, -outer_h/2),
            (outer_w/2, -outer_h/2),
            (outer_w/2, outer_h/2),
            (-outer_w/2, outer_h/2)
        ]
        
        # Inner square (counter-clockwise to create hole)
        inner_w = outer_w - 2 * wall_t
        inner_h = outer_h - 2 * wall_t
        inner_points = [
            (-inner_w/2, -inner_h/2),
            (-inner_w/2, inner_h/2),
            (inner_w/2, inner_h/2),
            (inner_w/2, -inner_h/2)
        ]
        
        # Combine outer and inner (for now, just use outer - hollow tubes are complex)
        # For simplicity, we'll create a solid tube and note this limitation
        return Geometry.from_prism(outer_points, self.params['length'])