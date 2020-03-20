import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from .load_data import csv2numpy

def play_video(video, win_name="testing"):
    """
    Play the video
    """

    # Open the video
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("Could not open the video, {}".format(video))

    # Play the video
    while True:
        ok, frame = cap.read()

        # Check if we have reached the end of the video
        if not ok: break

        # Show the frame
        cv2.imshow(win_name, frame)

        # Exit if ESC pressed
        key = cv2.waitKey(1) & 0xff
        if key == 27: break

    # Release the resource
    cap.release()
    cv2.destroyWindow(win_name)


def play_frames(frames, win_name="testing", sleep_time=None):
    """
    Play the frames
    """

    for frame in frames:
        cv2.imshow(win_name, frame)

        # Sleep
        if sleep_time is None:
            t = 0.0
        time.sleep(sleep_time)

        # Exit if ESC pressed
        key = cv2.waitKey(1) & 0xff
        if key == 27: break

    # Destroy the playing window
    cv2.destroyWindow(win_name)


def show_frames(frames_in, color_seq='BGR', figsize=(5, 5)):
    """
    Show the video.

    figsize: tuple
        Size of a single frame.
    """
    num_frames = len(frames_in)

    frame0 = frames_in[0]
    height, width, channels = frame0.shape

    # Plot settings
    width_plot = figsize[0]
    height_plot = figsize[1]*num_frames
    figsize_plot = (width_plot, height_plot)

    frames = []
    if channels != 1 and (color_seq == 'BGR' or color_seq == 'bgr'): # BGR format
        for i in range(num_frames):
            frame = cv2.cvtColor(frames_in[i], cv2.COLOR_BGR2RGB)
            frames.append(frame)
    else: # RGB or Gray format
        frames = frames_in

    fig, axes = plt.subplots(nrows=num_frames, ncols=1, figsize=figsize_plot)
    for i in range(num_frames):
        frame = frames[i]
        if channels == 1:
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


def frames2video(frames, video_path='output.mp4', codec=None):
    """
    Convert frames into video.

    Parameters
    ----------
    frames: List
        List of frames.
    video_path: string
        Video path.
    codec: string
        Codec; Available candidates: 'MJPG'.
    """

    num_frames = len(frames)
    win_name = 'video'

    # Frame information
    height, width, channels = frames[0].shape

    # Define the codec and create VideoWriter object
    if codec is None: codec = 'MJPG'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))

    print("Start to output {}".format(video_path))
    print("There are {} frames in total.".format(num_frames))

    for frame in frames:

        out.write(frame) # Write out frame to video
        cv2.imshow(win_name,frame)

        # Exit if ESC pressed
        key = cv2.waitKey(1) & 0xff
        if key == 27: break


    # Release everything if job is finished
    out.release()
    cv2.destroyWindow(win_name)


def load_video_csv(file_path, rows, cols, channels=1, start_col=0, header='infer'):
    """
    Load video file with csv format.

    Returns
    -------
    out: array
        Array of video, out[timesteps, rows, cols, channels]
    """

    arr_read = csv2numpy(file_path, start_col=start_col, header=header)
    timesteps = len(arr_read)
    out = np.zeros((timesteps, rows, cols, channels), dtype=np.int32)

    # Prepare the output array
    for ts in range(timesteps):
        for row in range(rows):
            for col in range(cols):
                for ch in range(channels):
                     out[ts, row, col, ch] = arr_read[ts, row*cols*ch + col*ch + ch]

    return out

