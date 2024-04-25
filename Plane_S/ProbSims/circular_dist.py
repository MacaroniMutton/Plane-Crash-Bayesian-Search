import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def makeCircularUniform(size, radius, center=None):
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    # Calculate distances from the center
    distances = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    # Create circular mask
    circular_mask = (distances <= radius).astype(float)

    return circular_mask

dist = makeCircularUniform(96, 30)
sns.heatmap(dist)
plt.show()



