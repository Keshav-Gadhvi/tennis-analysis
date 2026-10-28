# ==============================================================================
# UNIT CONVERSION UTILITIES
# ------------------------------------------------------------------------------
# These math helper functions use ratio proportion scaling to convert between 
# 2D screen pixels and real-world meters using known physical reference values
# (like court line width or player height).
# ==============================================================================

def convert_pixel_distance_to_meters(pixel_distance, refrence_height_in_meters, refrence_height_in_pixels):
    """
    Converts a distance measured in screen pixels to real-world meters.
    Formula: meters = (pixel_distance * real_meters) / reference_pixels
    """
    return (pixel_distance * refrence_height_in_meters) / refrence_height_in_pixels

def convert_meters_to_pixel_distance(meters, refrence_height_in_meters, refrence_height_in_pixels):
    """
    Converts a real-world distance in meters to screen pixel distance.
    Formula: pixels = (meters * reference_pixels) / real_meters
    """
    return (meters * refrence_height_in_pixels) / refrence_height_in_meters