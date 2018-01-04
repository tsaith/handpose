import numpy as np
import cv2 # OpenCV

def get_bbox_center(frame_width, frame_height, bbox_width, bbox_height):
    """
    Return the bbox at center.
    """
    xc = 0.5 + 0.5*frame_width
    yc = 0.5 + 0.5*frame_height

    x = round(xc - 0.5*bbox_width)
    y = round(yc - 0.5*bbox_height)

    bbox = (x, y, bbox_width, bbox_height)

    return bbox


def broaden_stroke(image, broaden_cells=2, threshold=0):
    # Broaden the stroke within an image.

    rows, cols = image.shape

    # Records the coordinates of pixels
    coordinates = []
    for i in range(rows):
        for j in range(cols):
            if image[i, j] > threshold: # Brodening
                coordinates.append((i, j))

    # Broadening
    shift = int(broaden_cells/2)
    print(shift)
    for i, j in coordinates:
        value = image[i, j]
        image[i, j-shift:j+shift+1] = value
        image[i-shift:i+shift+1, j] = value

    return image


def track_to_image(x_arr, y_arr, shape=(28, 28), broaden_cells=0):
    """
    Convert the 2D track into an image.

    Parameters
    ----------
    x_arr: array
        The x array.
    y_arr: array
        The y array.
    shape: tuple
        The shape of image whose shape is assumed as square.
    broaden_cells: int
        Number of cells to be broadened.

    Reurns
    ------
    image: array
        Image.
    """

    width, height = shape

    # Cells of margin
    len_margin = 3

    # Number of data points
    num = len(x_arr)

    # Data coordinates
    x_min = x_arr.min()
    x_max = x_arr.max()
    x_mean = 0.5*(x_max + x_min)

    y_min = y_arr.min()
    y_max = y_arr.max()
    y_mean = 0.5*(y_max + y_min)

    len_x = x_max - x_min
    len_y = y_max - y_min
    len_max = max(len_x, len_y)

    # Scale
    scale = 1.0*(width-2*len_margin)/len_max

    # Image coordinates in unit of pixel
    xp_mean = 0.5*width
    yp_mean = 0.5*height

    xp = (x_arr - x_mean) * scale + xp_mean
    yp = (y_arr - y_mean) * scale + yp_mean

    xp = xp.astype(int)
    yp = yp.astype(int)

    # Create an image
    image = np.zeros((width, height), dtype=np.uint8)
    for i in range(num):
        image[xp[i], yp[i]] = 255

    # Broaden the stroke
    broaden_stroke(image, broaden_cells=broaden_cells)

    return image

