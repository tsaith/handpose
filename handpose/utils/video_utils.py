import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_frames(frames):
    """
    Show the video.
    """
    num_frames = len(frames)
    rows, cols, channels = frames[0].shape

    fig, axes = plt.subplots(nrows=num_frames, ncols=1, figsize=(24, 24))

    for i in range(num_frames):

        frame = frames[i]
        if channels == 1:
            axes[i].imshow(frame, interpolation='nearest', cmap=plt.cm.gray)
        else:
            axes[i].imshow(frame, interpolation='nearest')

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
