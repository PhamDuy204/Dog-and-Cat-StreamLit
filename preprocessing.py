import skimage
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage import data, color,exposure
from skimage.transform import resize
def preprocessing(file):
    image = color.rgb2gray(np.array(Image.open(file).convert('RGB')))
    h,w =image.shape
    resized_img = resize(
        image, (128,256)
    )
    fd, hog_image = hog(
        resized_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
        visualize=True,
    )
    hog_image_rescaled = exposure.rescale_intensity(hog_image)
    return fd,resize(hog_image_rescaled,(h,w))