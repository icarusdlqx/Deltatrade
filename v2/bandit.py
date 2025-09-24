from __future__ import annotations
from typing import Dict

def sleeve_weights(defaults: Dict[str,float]) -> Dict[str,float]:
    s = sum(max(0.0, w) for w in defaults.values())
    if s <= 0: return {k: 0.0 for k in defaults}
    return {k: max(0.0, w)/s for k, w in defaults.items()}
