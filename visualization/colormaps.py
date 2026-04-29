from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def get_default_cmaps():
    good_colors = [
        [1, 0, 0],  # red
        [0, 1, 0],  # green
        [0, 0, 1],  # blue
        [1, 1, 0],  # yellow
        [1, 0, 1],  # magenta
        [255 / 255, 165 / 255, 0 / 255],  # orange
    ]

    # Add new, distinct colors
    new_colors = [
        [0 / 255, 255 / 255, 255 / 255],  # cyan
        [128 / 255, 0 / 255, 128 / 255],  # purple
        [0 / 255, 128 / 255, 128 / 255],  # teal
        [230 / 255, 190 / 255, 255 / 255],  # lavender
        [170 / 255, 110 / 255, 40 / 255],  # a good brown
        [255 / 255, 20 / 255, 147 / 255],  # deep pink
        [128 / 255, 128 / 255, 0 / 255],  # olive
        [0 / 255, 0 / 255, 128 / 255],  # navy
        [0 / 255, 100 / 255, 0 / 255],  # forest green
        [100 / 255, 200 / 255, 255 / 255],  # light blue
        [255 / 255, 100 / 255, 0 / 255],  # red-orange
        [130 / 255, 200 / 255, 25 / 255],  # lime green
        [255 / 255, 0 / 255, 100 / 255],  # pink
        [25 / 255, 100 / 255, 200 / 255], # cornflower blue

    ]

    all_colors = good_colors + new_colors

    cmaps = [CustomCmap([1, 1, 1], x) for x in all_colors]
    colormaps = [convert_cmap_to_cv(x) for x in cmaps]

    return colormaps


def CustomCmap(from_rgb, to_rgb):
    # from color r,g,b
    r1, g1, b1 = from_rgb

    # to color r,g,b
    r2, g2, b2 = to_rgb

    cdict = {'red': ((0, r1, r1),
                     (1, r2, r2)),
             'green': ((0, g1, g1),
                       (1, g2, g2)),
             'blue': ((0, b1, b1),
                      (1, b2, b2))}

    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap

def convert_cmap_to_cv(cmap, log=True, gamma=1.5):
    sm = plt.cm.ScalarMappable(cmap=cmap)
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    if log:
        color_range = (color_range ** gamma) / (255 ** gamma)
        color_range = np.uint8(color_range * 255)

    return color_range.reshape(256, 1, 3)
