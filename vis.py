import numpy as np
from matplotlib.colors import ListedColormap


def ndvi_vis(image, cloud_mask):
    palettes = {
        "RedGreen": [
            "#C03333",
            "#FF3333",
            "#FE7C33",
            "#FFC034",
            "#FFE41C",
            "#FEF913",
            "#EBEA48",
            "#E9F96A",
            "#C0F873",
            "#A8E74B",
            "#8FD93C",
            "#7FC23A",
            "#6AAE38",
            "#668E25",
            "#407520",
            "#2A5F17",
        ]
    }
    colors = palettes["RedGreen"]
    cloud_color = [178, 186, 182, 1]
    colormap = ListedColormap(name="custom_cmap", colors=colors)

    vis_lin_img = np.uint8(colormap(image) * 255)
    vis_lin_img[cloud_mask == 255] = cloud_color
    return vis_lin_img
