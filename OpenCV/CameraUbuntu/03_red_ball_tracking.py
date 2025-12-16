"""
Red ball tracking script using Astra camera
Detects and tracks a red ball in real-time using color segmentation
"""

from openni import openni2
import numpy as np
import cv2

# Initialize OpenNI2
openni2.initialize("/home/thom/saraJvB-main/OpenCV/CameraUbuntu/AstraSDK-v2.1.3/lib/Plugins/openni2/")

# Open the device (ASTRA camera)
dev = openni2.Device.open_any()

# Create a color stream
color_stream = dev.create_color_stream()
color_stream.start()

# Define range for red color in HSV
# Red is tricky in HSV because it wraps around, so we need two ranges
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Morphological kernel for noise reduction
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Ball properties
ball_center = None
ball_radius = 0
ball_velocity = np.array([0.0, 0.0])

try:
    while True:
        # Read a frame from the color stream
        color_frame = color_stream.read_frame()
        color_data = color_frame.get_buffer_as_uint8()

        # Convert the color data to a NumPy array
        color_array = np.ctypeslib.as_array(color_data)

        # Reshape to match frame dimensions (height, width, 3 channels)
        frame = color_array.reshape((color_frame.height, color_frame.width, 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 1)

        # Convert BGR to HSV for better red detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations to reduce noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process the largest contour (assuming it's the ball)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # Only process if the contour is large enough
            if area > 100:
                # Fit a circle to the contour
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(x), int(y))
                radius = int(radius)

                # Calculate velocity (simple difference from previous position)
                if ball_center is not None:
                    ball_velocity = np.array([x - ball_center[0], y - ball_center[1]])

                ball_center = center
                ball_radius = radius

                # Draw the circle
                cv2.circle(frame, center, radius, (0, 0, 255), 2)
                cv2.circle(frame, center, 5, (0, 255, 0), -1)  # Center point

                # Get contour moments for more accurate tracking
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 3, (255, 255, 0), -1)

                # Display ball information
                cv2.putText(frame, f"Ball Center: ({center[0]}, {center[1]})", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Radius: {radius}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Area: {area:.0f}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Velocity: ({ball_velocity[0]:.1f}, {ball_velocity[1]:.1f})", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw velocity vector
                if np.linalg.norm(ball_velocity) > 1:
                    end_point = (int(center[0] + ball_velocity[0] * 5), 
                                int(center[1] + ball_velocity[1] * 5))
                    cv2.arrowedLine(frame, center, end_point, (0, 255, 255), 2, tipLength=0.3)
            else:
                ball_center = None
                ball_radius = 0
        else:
            ball_center = None
            ball_radius = 0

        # Display instructions
        cv2.putText(frame, "Red Ball Tracking - Press 'q' to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the original frame and mask side by side
        display_frame = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow("Red Ball Tracking", display_frame)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Clean up
    color_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()
