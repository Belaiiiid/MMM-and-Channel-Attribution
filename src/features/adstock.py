import numpy as np

def geometric_adstock(x, decay: float):
    """
    Applies geometric adstock transformation.

    Parameters:
    - x: array-like media spend
    - decay: carryover rate (0 < decay < 1)

    Returns:
    - adstocked series
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)

    for t in range(len(x)):
        out[t] = x[t] + (decay * out[t - 1] if t > 0 else 0.0)

    return out
