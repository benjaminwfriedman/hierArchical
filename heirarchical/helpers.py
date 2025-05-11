import random
import uuid 

def generate_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def random_color(seed=None):
    if seed is not None:
        random.seed(seed)
    r = random.randint(0, 250)
    g = random.randint(0, 250)
    b = random.randint(0, 250)
    return f"rgb({r},{g},{b})"

def normalize_ifc_enum(value: str) -> str:
    """
    Normalize IFC-style enum strings (e.g., '.SINGLE_SWING_LEFT.') to lowercase underscore format.
    """
    if not value:
        return ""
    return value.strip('.').lower()