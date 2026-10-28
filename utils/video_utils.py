# ==============================================================================
# VIDEO INPUT / OUTPUT UTILITIES
# ------------------------------------------------------------------------------
# Functions for reading an input MP4 video file frame-by-frame into memory, and
# writing/saving annotated output video files back to disk using OpenCV VideoWriter.
# ==============================================================================

import cv2

def read_video(video_path):
    """
    Reads a video file from disk and returns a list containing all image frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    """
    Writes a list of image frames into an output AVI video file at 24 frames per second.
    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()