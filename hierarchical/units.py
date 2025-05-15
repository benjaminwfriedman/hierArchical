from enum import Enum


class UnitSystem(Enum):
    """Common unit systems used in architecture/construction."""
    MILLIMETER = "mm"
    CENTIMETER = "cm" 
    METER = "m"
    KILOMETER = "km"
    INCH = "in"
    FOOT = "ft"
    YARD = "yd"
    MILE = "mi"

# Conversion factors to meters (base unit)
UNIT_TO_METER = {
    UnitSystem.MILLIMETER: 0.001,
    UnitSystem.CENTIMETER: 0.01,
    UnitSystem.METER: 1.0,
    UnitSystem.KILOMETER: 1000.0,
    UnitSystem.INCH: 0.0254,
    UnitSystem.FOOT: 0.3048,
    UnitSystem.YARD: 0.9144,
    UnitSystem.MILE: 1609.344,
}

# Common unit systems in AEC
class UnitSystems:
    METRIC = {UnitSystem.MILLIMETER, UnitSystem.CENTIMETER, UnitSystem.METER, UnitSystem.KILOMETER}
    IMPERIAL = {UnitSystem.INCH, UnitSystem.FOOT, UnitSystem.YARD, UnitSystem.MILE}