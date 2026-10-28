# ==============================================================================
# BOUNDING BOX & GEOMETRY UTILITIES
# ------------------------------------------------------------------------------
# Helper functions for geometry math: calculating bounding box centers, player 
# feet locations (bottom-center of bounding box), Euclidean distances, box 
# height in pixels, and finding nearest court keypoints.
# ==============================================================================

def get_center_of_bbox(bbox):
    """
    Calculates the center (x, y) coordinates of a bounding box [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def measure_distance(p1, p2):
    """
    Calculates 2D Euclidean distance between two points (p1, p2) using Pythagorean theorem.
    Formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def get_foot_position(bbox):
    """
    Returns the bottom-center coordinate of a player's bounding box, representing
    where their feet touch the court ground.
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int(y2))

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    """
    Finds which court line keypoint in keypoints list is vertically closest to a point.
    """
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    for keypoint_indix in keypoint_indices:
        keypoint = keypoints[keypoint_indix * 2], keypoints[keypoint_indix * 2 + 1]
        distance = abs(point[1] - keypoint[1])

        if distance < closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_indix
     
    return key_point_ind

def get_height_of_bbox(bbox):
    """
    Calculates the height of a bounding box in pixels (y2 - y1).
    """
    return bbox[3] - bbox[1]

def measure_xy_distance(p1, p2):
    """
    Calculates absolute delta X and delta Y distances between two points.
    """
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])