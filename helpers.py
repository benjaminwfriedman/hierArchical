import random
import uuid 

def generate_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def random_color(seed=None):
    if seed is not None:
        random.seed(seed)
    r = random.randint(50, 200)
    g = random.randint(50, 200)
    b = random.randint(50, 200)
    return f"rgb({r},{g},{b})"