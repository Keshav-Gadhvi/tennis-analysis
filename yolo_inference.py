# ==============================================================================
# STANDALONE YOLO INFERENCE DEMO SCRIPT
# ------------------------------------------------------------------------------
# Layman Explanation:
# A simple standalone test script to run YOLOv8 object tracking directly on an
# input video file to test detection capabilities before running the full pipeline.
# ==============================================================================

from ultralytics import YOLO 

# Load the pretrained YOLOv8 extra-large (yolov8x) model
model = YOLO('yolov8x')

# Run tracking on input video with confidence threshold of 0.20 and save annotated result
result = model.track('input_videos/input_video.mp4', conf=0.2, save=True)

# Print detected bounding box results for debugging
# print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)