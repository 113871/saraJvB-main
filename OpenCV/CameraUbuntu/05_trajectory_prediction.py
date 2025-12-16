"""
3D Red ball tracking with trajectory prediction using physics
Combines 3D tracking with gravity-based projectile motion prediction
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

# Astra camera intrinsic parameters
focal_length_x = 570.0
focal_length_y = 570.0
principal_point_x = 320.0
principal_point_y = 240.0

# Physics parameters
GRAVITY = 9810.0  # mm/s² (acceleration due to gravity)
PREDICTION_FRAMES = 30  # Number of frames to predict ahead
TIME_STEP = 0.033  # Approximate time between frames in seconds (30 FPS)

# Define range for red color in HSV
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Morphological kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def pixel_to_3d(x_pixel, y_pixel, depth_mm):
    """Convert 2D pixel coordinates and depth to 3D world coordinates."""
    if depth_mm == 0:
        return None
    
    x_normalized = (x_pixel - principal_point_x) / focal_length_x
    y_normalized = (y_pixel - principal_point_y) / focal_length_y
    
    X = x_normalized * depth_mm
    Y = y_normalized * depth_mm
    Z = depth_mm
    
    return np.array([X, Y, Z])

def project_3d_to_2d(X, Y, Z):
    """Convert 3D world coordinates back to 2D pixel coordinates."""
    if Z == 0:
        return None
    
    x_pixel = (X / Z) * focal_length_x + principal_point_x
    y_pixel = (Y / Z) * focal_length_y + principal_point_y
    
    return (int(x_pixel), int(y_pixel))

def predict_trajectory(position, velocity, num_frames, time_step):
    """
    Predict ball trajectory using physics (projectile motion with gravity).
    
    Args:
        position: Current 3D position [X, Y, Z] in mm
        velocity: Current 3D velocity [Vx, Vy, Vz] in mm/frame
        num_frames: Number of frames to predict
        time_step: Time between frames in seconds
        
    Returns:
        List of predicted 3D positions
    """
    trajectory = [position.copy()]
    current_pos = position.copy()
    current_vel = velocity.copy()
    
    # Convert velocity from mm/frame to mm/s
    vel_per_second = velocity / time_step
    
    for frame in range(num_frames):
        # Update velocity (gravity acts downward in Z direction)
        # Note: In camera coordinates, Z is forward, so gravity affects the trajectory
        vel_per_second[2] += GRAVITY * time_step
        
        # Update position
        current_pos = current_pos + (vel_per_second * time_step)
        
        trajectory.append(current_pos.copy())
    
    return trajectory

# Tracking variables
position_history = []
velocity_3d = np.array([0.0, 0.0, 0.0])
ball_center_3d = None
prev_center_3d = None
trajectory_3d = []

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

        # Detect red ball
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
                        velocity_3d = ball_center_3d - prev_center_3d
                    
                    prev_center_3d = ball_center_3d
                    position_history.append(ball_center_3d.copy())
                    
                    # Keep only last 10 positions for smoother velocity calculation
                    if len(position_history) > 10:
                        position_history.pop(0)

                    # Draw current ball position
                    cv2.circle(frame, center_2d, radius, (0, 0, 255), 2)
                    cv2.circle(frame, center_2d, 5, (0, 255, 0), -1)

                    # Predict trajectory if we have enough velocity data
                    if np.linalg.norm(velocity_3d) > 0.1:  # Only predict if ball is moving
                        trajectory_3d = predict_trajectory(
                            ball_center_3d, 
                            velocity_3d, 
                            PREDICTION_FRAMES, 
                            TIME_STEP
                        )
                        
                        # Draw predicted trajectory on 2D frame
                        prev_2d = center_2d
                        for i, pos_3d in enumerate(trajectory_3d[1:]):
                            # Only draw if point is in front of camera and within reasonable depth
                            if 100 < pos_3d[2] < 3000:
                                point_2d = project_3d_to_2d(pos_3d[0], pos_3d[1], pos_3d[2])
                                if point_2d and 0 < point_2d[0] < frame.shape[1] and 0 < point_2d[1] < frame.shape[0]:
                                    # Color changes from cyan to red as time goes on
                                    color = (255 - int(i * 8), 255, int(i * 8))
                                    cv2.circle(frame, point_2d, 2, color, -1)
                                    if i > 0:
                                        cv2.line(frame, prev_2d, point_2d, color, 1)
                                    prev_2d = point_2d
                        
                        # Find landing point (where Z reaches ground level, assuming camera is ~500mm above ground)
                        landing_point = None
                        for pos_3d in trajectory_3d:
                            if pos_3d[2] > 3000:  # Ball has gone beyond far field
                                landing_point = pos_3d
                                break
                        
                        if landing_point:
                            landing_2d = project_3d_to_2d(landing_point[0], landing_point[1], landing_point[2])
                            if landing_2d and 0 < landing_2d[0] < frame.shape[1] and 0 < landing_2d[1] < frame.shape[0]:
                                cv2.circle(frame, landing_2d, 10, (0, 0, 255), 2)
                                cv2.putText(frame, "Landing", (landing_2d[0] + 10, landing_2d[1]),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # Display information
                    cv2.putText(frame, f"3D Position (mm):", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(frame, f"  X: {ball_center_3d[0]:.1f}  Y: {ball_center_3d[1]:.1f}  Z: {ball_center_3d[2]:.1f}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    cv2.putText(frame, f"3D Velocity (mm/frame):", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"  Vx: {velocity_3d[0]:.1f}  Vy: {velocity_3d[1]:.1f}  Vz: {velocity_3d[2]:.1f}", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    speed_3d = np.linalg.norm(velocity_3d)
                    cv2.putText(frame, f"Speed: {speed_3d:.1f} mm/frame", (10, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Display legend
        cv2.putText(frame, "Trajectory Prediction with Physics", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Cyan -> Red: Predicted path", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show frame
        cv2.imshow("Ball Trajectory Prediction", frame)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Clean up
    color_stream.stop()
    depth_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()
