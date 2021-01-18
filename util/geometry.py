import numpy as np

def check_inside_sliced_unit_circle(x, y, start_angle, end_angle):
    # Rotate angles from "clockwise North" to the standard "anti-clockwise East"
    start_angle, end_angle = (-end_angle + 90) % 360, (-start_angle + 90) % 360

    # Radius
    radius = np.sqrt(x ** 2 + y ** 2)

    # Angle in degrees
    angle = np.arctan2(y, x)
    angle = np.rad2deg(angle) % 360

    if radius <= 1:
        if start_angle <= angle <= end_angle:
            return True
        elif start_angle > end_angle and (angle > start_angle or angle < end_angle):
            return True
    return False