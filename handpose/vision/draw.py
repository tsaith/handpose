import cv2

def draw_rect(image, rect, color=(0, 255, 0), thickness=2):
    """
    Draw a rectangle.
    """

    # Starting and ending point of the rectangle
    p0 = (rect.x, rect.y)
    p1 = (rect.x + rect.width, rect.y + rect.height)

    cv2.rectangle(image, p0, p1, color, thickness)


def open_webcam(win_name="demo"):
    """
    Open the webcam.
    Hit the ESC key to exit.
    """

    cv2.namedWindow(win_name)
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow(win_name, frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow(win_name)
