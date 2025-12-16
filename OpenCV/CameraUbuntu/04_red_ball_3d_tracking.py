"""
3D Red ball tracking script using Astra camera
Combines color and depth streams to get 3D position of the ball
"""

from openni import openni2
import numpy as np
import cv2

# Initialize OpenNI2
openni2.initialize("/home/thom/saraJvB-main/OpenCV/CameraUbuntu/AstraSDK-v2.1.3/lib/Plugins/openni2/")

# Open the device (ASTRA camera)
dev = openni2.Device.open_any()

# Create color and depth streams
color_stream = dev.create_color_stream()
depth_stream = dev.create_depth_stream()
color_stream.start()
depth_stream.start()

# Astra camera intrinsic parameters (typical values - adjust if needed)
# These are focal lengths and principal point
focal_length_x = 570.0  # fx in pixels
focal_length_y = 570.0  # fy in pixels
principal_point_x = 320.0  # cx (image center x)
principal_point_y = 240.0  # cy (image center y)

# Define range for red color in HSV
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Morphological kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def pixel_to_3d(x_pixel, y_pixel, depth_mm):
    """
    Convert 2D pixel coordinates and depth to 3D world coordinates.
    
    Args:
        x_pixel: X coordinate in pixels
        y_pixel: Y coordinate in pixels
        depth_mm: Depth in millimeters
        
    Returns:
        (X, Y, Z) in millimeters
    """
    # Avoid division by zero
    if depth_mm == 0:
        return None
    
    # Convert pixel coordinates to normalized image coordinates
    x_normalized = (x_pixel - principal_point_x) / focal_length_x
    y_normalized = (y_pixel - principal_point_y) / focal_length_y
    
    # Convert to 3D world coordinates
    X = x_normalized * depth_mm
    Y = y_normalized * depth_mm
    Z = depth_mm
    
    return (X, Y, Z)

ball_center_3d = None
prev_center_3d = None
velocity_3d = np.array([0.0, 0.0, 0.0])

try:
    while True:
        # Read frames
        color_frame = color_stream.read_frame()
        depth_frame = depth_stream.read_frame()
        
        # Process color frame
        color_data = color_frame.get_buffer_as_uint8()
        color_array = np.ctypeslib.as_array(color_data)
        frame = color_array.reshape((color_frame.height, color_frame.width, 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 1)

        # Process depth frame
        depth_data = depth_frame.get_buffer_as_uint16()
        depth_array = np.ctypeslib.as_array(depth_data)
        depth_image = depth_array.reshape((depth_frame.height, depth_frame.width))
        depth_image = cv2.flip(depth_image, 1)

        # Detect red ball in color frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 100:
                # Get ball center in 2D
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                center_2d = (int(x), int(y))
                radius = int(radius)

                # Get depth at ball center
                depth_at_center = depth_image[center_2d[1], center_2d[0]]

                # Convert to 3D coordinates
                if depth_at_center > 0:
                    ball_center_3d = pixel_to_3d(center_2d[0], center_2d[1], depth_at_center)
                    
                    # Calculate 3D velocity
                    if prev_center_3d is not None:
                        velocity_3d = np.array([
                            ball_center_3d[0] - prev_center_3d[0],
                            ball_center_3d[1] - prev_center_3d[1],
                            ball_center_3d[2] - prev_center_3d[2]
                        ])
                    
                    prev_center_3d = ball_center_3d

                    # Draw on frame
                    cv2.circle(frame, center_2d, radius, (0, 0, 255), 2)
                    cv2.circle(frame, center_2d, 5, (0, 255, 0), -1)

                    # Display 2D information
                    cv2.putText(frame, f"2D Center: ({center_2d[0]}, {center_2d[1]})", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Radius: {radius}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Display 3D information
                    cv2.putText(frame, f"3D Position (mm):", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(frame, f"  X: {ball_center_3d[0]:.1f}", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(frame, f"  Y: {ball_center_3d[1]:.1f}", (10, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(frame, f"  Z: {ball_center_3d[2]:.1f}", (10, 210),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # Display 3D velocity
                    speed_3d = np.linalg.norm(velocity_3d)
                    cv2.putText(frame, f"3D Velocity (mm/frame):", (10, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"  X: {velocity_3d[0]:.1f}", (10, 270),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(frame, f"  Y: {velocity_3d[1]:.1f}", (10, 300),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(frame, f"  Z: {velocity_3d[2]:.1f}", (10, 330),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(frame, f"  Speed: {speed_3d:.1f} mm/frame", (10, 360),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # Display depth
                    cv2.putText(frame, f"Depth: {depth_at_center} mm", (10, 390),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Display instructions
        cv2.putText(frame, "3D Ball Tracking - Press 'q' to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Create depth visualization
        depth_display = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        # Show frames
        cv2.imshow("3D Ball Tracking - Color", frame)
        cv2.imshow("3D Ball Tracking - Depth", depth_display)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Clean up
    color_stream.stop()
    depth_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()
