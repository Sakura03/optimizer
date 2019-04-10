import numpy as np
from PIL import Image
import os, sys
from os.path import join, split, splitext, isdir, isfile, abspath

__img_ext__ = ['jpg', 'jpeg', 'png', 'bmp']

def isimg(path):
    assert isfile(path)
    path = path.lower()
    return any(path.endswith(ext) for ext in __img_ext__)

def traverse_images(path):
    """
    traverse a given folder for images
    input:  path to traverse for images
    return: a list containing all image paths
    """
    images = []
    for p in os.listdir(path):
        p1 = abspath(join(path, p))
        if isdir(p1):
            images += traverse_images(p1)
        elif isimg(p1):
            images.append(p1)
        else:
            continue

    return images

def gray2rgb(x):
    if x.ndim == 3 and x.shape[2] ==3:
        return x
    elif x.ndim == 2:
        H, W = x.shape
        x = x.reshape((H, W, 1))
        x = np.repeat(x, 3, 2)
        return x
    else:
        raise ValueError("Unsupported shape: %s" % str(x.shape))

def imread(path):
    assert isfile(path), "%s does't exist!" % path
    assert isimg(path), "%s seems not an image!" % path
    return np.array(Image.open(path))

def norm01(x):
    """
    Normalize input image into range [0, 1]
    """
    assert isinstance(x, np.ndarray)
    x = x.astype(np.float32)

    if x.max() == x.min():
        return x - x.min()

    x -= x.min()
    x /= x.max()

    return x

def norm255(x):
    """
    Normalize input image into range [0, 1]
    """
    assert isinstance(x, np.ndarray)
    x = x.astype(np.float32)

    if x.max() == x.min():
        x[...] = 0
        return x
    x = norm01(x)

    return (x * 255).astype(np.uint8)

def hwc2nchw(image):
    """
    Convert an image (or a list of images of the same size) to
    [N C H W] style batch data
    
    Input:
        --- image: numpy.ndarray or list of numpy.ndarray
    """
    assert isinstance(image, np.ndarray) or isinstance(image, list)
    H, W, C, N = 0, 0, 0, 0
    if isinstance(image, np.ndarray):
        assert image.ndim == 3
        H, W, C = image.shape
        N = 1
    else:
        for i, im in enumerate(image):
            assert im.ndim == 3
            assert isinstance(im, np.ndarray)
            if i == 0:
                pre_shape = im.shape
            else:
                assert im.shape == pre_shape
        H, W, C = image[0].shape
        N = len(image)
    data = np.zeros((N, C, H, W), dtype=np.float32)

    if isinstance(image, np.ndarray):
        data[0, :, :, :] = image.transpose((2, 0, 1))
    else:
        for i, im in enumerate(image):
            data[i, :, :, :] = im.transpose((2, 0, 1))

    return data