
import os
from datetime import datetime

def make_output_dir(base_dir: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base_dir, ts)
    os.makedirs(out, exist_ok=True)
    return out
