import numpy as np
import matplotlib.pyplot as plt
import cv2


def color2gray(color, color_seq="BGR", has_channel=True):
    """
    Convert color image as gray one.
    """

    if color_seq == "BGR":
        convert_type = cv2.COLOR_BGR2GRAY
    else:
        convert_type = cv2.COLOR_RGB2GRAY

    #rows, cols, channels = color.shape
    #gray = np.zeros((rows, cols, 1), dtype=np.uint8)
    gray = cv2.cvtColor(color, convert_type)
    #gray[:, :, 0] = int(np.mean(color, keepdims=True))
    if has_channel:
        gray = np.expand_dims(gray, axis=2)

    return gray

def show_frames(frames_in, color_seq='RGB', figsize=(16, 16)):
    """
    Show the video.
    """
    num_frames = len(frames_in)

    frame0 = frames_in[0]
    rows, cols, channels = frame0.shape

    frames = []
    if channels != 1 and (color_seq == 'BGR' or color_seq == 'bgr'): # BGR format
        for i in range(num_frames):
            frame = cv2.cvtColor(frames_in[i], cv2.COLOR_BGR2RGB)
            frames.append(frame)
    else: # RGB or Gray format
        frames = frames_in

    fig, axes = plt.subplots(nrows=num_frames, ncols=1, figsize=figsize)
    for i in range(num_frames):
        frame = frames[i]
        if channels == 1:
            print("shape = ", frame.shape)
            axes[i].imshow(frame[:,:,0], interpolation='nearest', cmap=plt.cm.gray)
        else:
            axes[i].imshow(frame, interpolation='nearest')

        title = "frame {}".format(i)
        axes[i].set_title(title)

    return fig, axes

def show_video(video):
    """
    Show the video.
    """
    timesteps, rows, cols, channels = video.shape

    fig, axes = plt.subplots(nrows=timesteps, ncols=1, figsize=(24, 24))

    for ts in range(timesteps):

        if channels == 1:
            image = video[ts, :, :, 0]
            axes[ts].imshow(image, interpolation='nearest', cmap=plt.cm.gray)
        else:
            image = video[ts]
            axes[ts].imshow(image, interpolation='nearest')

    return fig, axes

def video2frames(video):
    """
    Convert a video into frames.

    Parameters
    ----------
    video: string
        Video path.

    Returns
    -------
    frames: List
        Frames of video.
    """

    frames = []
    cap = cv2.VideoCapture(video)
    while True:
        grabbed, frame = cap.read()
        if grabbed:
            frames.append(frame)
        else:
            break
    cap.release()

    return frames
