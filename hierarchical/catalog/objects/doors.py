"""
Parametric door objects - complete door assemblies.
"""

from typing import Dict, List
from ..base import ParametricObject, Parameter
from ..elements.lumber import Lumber2X4, Lumber2X6
from hierarchical.items import Component


class SwingDoor(ParametricObject):
    """Standard swing door with frame and hardware"""
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'width': Parameter(
                name='width',
                type=float,
                default=3.0,
                min_value=2.0,
                max_value=4.0,
                unit='ft',
                description="Door width"
            ),
            'height': Parameter(
                name='height',
                type=float,
                default=6.67,
                min_value=6.0,
                max_value=8.0,
                unit='ft',
                description="Door height"
            ),
            'swing_direction': Parameter(
                name='swing_direction',
                type=str,
                default='inward_right',
                description="Swing direction (inward_right, inward_left, outward_right, outward_left)"
            ),
            'door_type': Parameter(
                name='door_type',
                type=str,
                default='interior',
                description="Door type (interior, exterior, pocket)"
            ),
            'material': Parameter(
                name='material',
                type=str,
                default='wood',
                description="Door material (wood, steel, fiberglass)"
            ),
            'frame_material': Parameter(
                name='frame_material',
                type=str,
                default='wood',
                description="Frame material (wood, steel)"
            )
        }
    
    def create_components(self) -> List[Component]:
        """Create door frame and door panel components"""
        components = []
        
        width = self.params['width']
        height = self.params['height']
        door_type = self.params['door_type']
        material = self.params['material']
        frame_material = self.params['frame_material']
        
        # Create door frame component
        frame = self._create_door_frame(width, height, frame_material)
        if frame:
            components.append(frame)
        
        # Create door panel component
        panel = self._create_door_panel(width, height, material)
        if panel:
            components.append(panel)
        
        # Create hardware component
        hardware = self._create_door_hardware(door_type)
        if hardware:
            components.append(hardware)
        
        return components
    
    def _create_door_frame(self, width: float, height: float, frame_material: str) -> Component:
        """Create door frame component"""
        from ..elements.lumber import Lumber2X4
        
        frame_elements = []
        
        # Frame dimensions (simplified - actual frames are complex)
        frame_thickness = 0.75 / 12  # 3/4" thick frame
        frame_width = 4.5 / 12       # 4.5" wide jamb
        
        # Create simplified frame elements
        # Top jamb
        top_jamb = Lumber2X4(length=width + frame_width, species=f"{frame_material}_jamb")
        top_jamb.name = f"{self.name}_top_jamb"
        top_jamb.move(dz=height - frame_thickness)
        frame_elements.append(top_jamb)
        
        # Side jambs
        left_jamb = Lumber2X4(length=height, species=f"{frame_material}_jamb")
        left_jamb.name = f"{self.name}_left_jamb"
        left_jamb.rotate_z(90)
        frame_elements.append(left_jamb)
        
        right_jamb = Lumber2X4(length=height, species=f"{frame_material}_jamb")
        right_jamb.name = f"{self.name}_right_jamb"
        right_jamb.move(dx=width + frame_width)
        right_jamb.rotate_z(90)
        frame_elements.append(right_jamb)
        
        return Component.from_elements(
            elements=tuple(frame_elements),
            name=f"{self.name}_frame",
            type="door_frame"
        )
    
    def _create_door_panel(self, width: float, height: float, material: str) -> Component:
        """Create door panel component"""
        from ..elements.lumber import Lumber2X4
        
        panel_elements = []
        
        # Simplified door panel as single element
        door_thickness = 1.75 / 12  # 1-3/4" thick door
        
        panel = Lumber2X4(length=width, species=f"{material}_door_panel")
        panel.name = f"{self.name}_panel"
        # Position in opening
        panel.move(dx=width/2, dy=door_thickness/2, dz=height/2)
        panel_elements.append(panel)
        
        return Component.from_elements(
            elements=tuple(panel_elements),
            name=f"{self.name}_panel",
            type="door_panel"
        )
    
    def _create_door_hardware(self, door_type: str) -> Component:
        """Create door hardware component"""
        from ..elements.lumber import Lumber2X4
        
        hardware_elements = []
        
        # Simplified hardware as small elements (using minimum allowed length)
        handle = Lumber2X4(length=0.5, species=f"{door_type}_handle")
        handle.name = f"{self.name}_handle"
        hardware_elements.append(handle)
        
        hinges = Lumber2X4(length=0.5, species=f"{door_type}_hinges")  # Minimum 0.5ft
        hinges.name = f"{self.name}_hinges"
        hardware_elements.append(hinges)
        
        return Component.from_elements(
            elements=tuple(hardware_elements),
            name=f"{self.name}_hardware",
            type="door_hardware"
        )


class SlidingDoor(ParametricObject):
    """Sliding door assembly"""
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'width': Parameter(
                name='width',
                type=float,
                default=6.0,
                min_value=4.0,
                max_value=12.0,
                unit='ft',
                description="Door width"
            ),
            'height': Parameter(
                name='height',
                type=float,
                default=6.67,
                min_value=6.0,
                max_value=8.0,
                unit='ft',
                description="Door height"
            ),
            'num_panels': Parameter(
                name='num_panels',
                type=int,
                default=2,
                min_value=1,
                max_value=4,
                description="Number of sliding panels"
            ),
            'material': Parameter(
                name='material',
                type=str,
                default='glass',
                description="Door material (glass, wood, aluminum)"
            ),
            'track_type': Parameter(
                name='track_type',
                type=str,
                default='standard',
                description="Track type (standard, heavy_duty, pocket)"
            )
        }
    
    def create_components(self) -> List[Component]:
        """Create sliding door components"""
        components = []
        
        width = self.params['width']
        height = self.params['height']
        num_panels = self.params['num_panels']
        material = self.params['material']
        
        # Create track component
        track = self._create_track_system(width, height)
        if track:
            components.append(track)
        
        # Create panels
        panel_width = width / num_panels
        for i in range(num_panels):
            panel = self._create_sliding_panel(panel_width, height, material, i)
            if panel:
                components.append(panel)
        
        return components
    
    def _create_track_system(self, width: float, height: float) -> Component:
        """Create track system component"""
        from ..elements.lumber import Lumber2X4
        
        track_elements = []
        
        # Top track
        top_track = Lumber2X4(length=width, species="aluminum_track")
        top_track.name = f"{self.name}_top_track"
        top_track.move(dz=height)
        track_elements.append(top_track)
        
        # Bottom track
        bottom_track = Lumber2X4(length=width, species="aluminum_track")
        bottom_track.name = f"{self.name}_bottom_track"
        track_elements.append(bottom_track)
        
        return Component.from_elements(
            elements=tuple(track_elements),
            name=f"{self.name}_track_system",
            type="sliding_track"
        )
    
    def _create_sliding_panel(self, width: float, height: float, material: str, panel_num: int) -> Component:
        """Create individual sliding panel"""
        from ..elements.lumber import Lumber2X4
        
        panel_elements = []
        
        panel = Lumber2X4(length=width, species=f"{material}_sliding_panel")
        panel.name = f"{self.name}_panel_{panel_num + 1}"
        # Position panel
        panel.move(dx=width * panel_num, dz=height/2)
        panel_elements.append(panel)
        
        return Component.from_elements(
            elements=tuple(panel_elements),
            name=f"{self.name}_panel_{panel_num + 1}",
            type="sliding_panel"
        )


class PocketDoor(ParametricObject):
    """Pocket door assembly that slides into wall"""
    
    @classmethod
    def get_parameters(cls) -> Dict[str, Parameter]:
        return {
            'width': Parameter(
                name='width',
                type=float,
                default=2.5,
                min_value=2.0,
                max_value=3.5,
                unit='ft',
                description="Door width"
            ),
            'height': Parameter(
                name='height',
                type=float,
                default=6.67,
                min_value=6.0,
                max_value=8.0,
                unit='ft',
                description="Door height"
            ),
            'wall_thickness': Parameter(
                name='wall_thickness',
                type=float,
                default=6.0,
                min_value=4.0,
                max_value=8.0,
                unit='in',
                description="Wall thickness to accommodate pocket"
            ),
            'material': Parameter(
                name='material',
                type=str,
                default='wood',
                description="Door material (wood, composite)"
            )
        }
    
    def create_components(self) -> List[Component]:
        """Create pocket door components"""
        components = []
        
        width = self.params['width']
        height = self.params['height']
        wall_thickness = self.params['wall_thickness'] / 12  # Convert to feet
        material = self.params['material']
        
        # Create pocket frame
        frame = self._create_pocket_frame(width, height, wall_thickness)
        if frame:
            components.append(frame)
        
        # Create door panel
        panel = self._create_pocket_panel(width, height, material)
        if panel:
            components.append(panel)
        
        # Create track hardware
        hardware = self._create_pocket_hardware(width)
        if hardware:
            components.append(hardware)
        
        return components
    
    def _create_pocket_frame(self, width: float, height: float, wall_thickness: float) -> Component:
        """Create pocket door frame"""
        from ..elements.lumber import Lumber2X6
        
        frame_elements = []
        
        # Header
        header = Lumber2X6(length=width * 2, species="pocket_header")  # Double width for pocket
        header.name = f"{self.name}_header"
        header.move(dz=height)
        frame_elements.append(header)
        
        # Pocket studs
        pocket_stud = Lumber2X6(length=height, species="pocket_stud")
        pocket_stud.name = f"{self.name}_pocket_stud"
        pocket_stud.move(dx=width)
        frame_elements.append(pocket_stud)
        
        return Component.from_elements(
            elements=tuple(frame_elements),
            name=f"{self.name}_pocket_frame",
            type="pocket_frame"
        )
    
    def _create_pocket_panel(self, width: float, height: float, material: str) -> Component:
        """Create pocket door panel"""
        from ..elements.lumber import Lumber2X4
        
        panel_elements = []
        
        panel = Lumber2X4(length=width, species=f"{material}_pocket_door")
        panel.name = f"{self.name}_pocket_panel"
        panel.move(dx=width/2, dz=height/2)
        panel_elements.append(panel)
        
        return Component.from_elements(
            elements=tuple(panel_elements),
            name=f"{self.name}_pocket_panel",
            type="pocket_panel"
        )
    
    def _create_pocket_hardware(self, width: float) -> Component:
        """Create pocket door hardware"""
        from ..elements.lumber import Lumber2X4
        
        hardware_elements = []
        
        # Ensure track length meets minimum requirement
        track_length = max(width * 2, 0.5)  # Minimum 0.5ft
        track = Lumber2X4(length=track_length, species="pocket_track")
        track.name = f"{self.name}_pocket_track"
        hardware_elements.append(track)
        
        return Component.from_elements(
            elements=tuple(hardware_elements),
            name=f"{self.name}_pocket_hardware",
            type="pocket_hardware"
        )