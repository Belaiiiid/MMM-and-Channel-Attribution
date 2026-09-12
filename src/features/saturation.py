import numpy as np

def hill_saturation(x, alpha: float, ec50: float):
    """
    Hill function for diminishing returns.

    Parameters:
    - x: adstocked media
    - alpha: curve steepness
    - ec50: half-saturation point

    Returns:
    - saturated response
    """
    x = np.maximum(x, 0)
    return (x ** alpha) / (x ** alpha + ec50 ** alpha)
