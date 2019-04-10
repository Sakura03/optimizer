import numpy as np

def rand_palette(num_classes=21):
    return (np.random.rand(num_classes, 3)*255).astype(np.uint8)

def color_encode(labelmap, colors=rand_palette(100)):

    assert isinstance(labelmap, np.ndarray)
    labelmap = np.squeeze(labelmap)
    assert labelmap.ndim == 2

    H, W = labelmap.shape
    colormap = np.zeros((H, W, 3))

    for lb in np.unique(labelmap):
        colormap[labelmap == lb, :] = colors[lb]

    return colormap.astype(np.uint8)
