"""
Hand tracking script using MediaPipe and Astra camera
Detects and tracks hands with finger landmarks and joint positions
"""

from openni import openni2
import numpy as np
import cv2
import mediapipe as mp

# Initialize MediaPipe Hand detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize OpenNI2
openni2.initialize("/home/thom/saraJvB-main/OpenCV/CameraUbuntu/AstraSDK-v2.1.3/lib/Plugins/openni2/")

# Open the device (ASTRA camera)
dev = openni2.Device.open_any()

# Create a color stream (hand tracking works with RGB frames)
color_stream = dev.create_color_stream()
color_stream.start()

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

        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for hand detection
        results = hands.process(frame_rgb)

        # Draw hand landmarks if detected
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw the hand skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Display hand label (Left or Right)
                hand_label = handedness.classification[0].label
                confidence = handedness.classification[0].score
                
                # Get the bounding box of the hand
                h, w, c = frame.shape
                landmarks_2d = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]
                
                # Find the top-left corner for text placement
                min_x = min([lm[0] for lm in landmarks_2d])
                min_y = min([lm[1] for lm in landmarks_2d])
                
                # Draw hand info
                text = f"{hand_label} ({confidence:.2f})"
                cv2.putText(frame, text, (int(min_x), int(min_y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display finger positions
                finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                finger_tips = [4, 8, 12, 16, 20]  # MediaPipe landmark indices for fingertips
                
                for i, (finger_name, tip_idx) in enumerate(zip(finger_names, finger_tips)):
                    tip_x = int(landmarks_2d[tip_idx][0])
                    tip_y = int(landmarks_2d[tip_idx][1])
                    cv2.putText(frame, finger_name, (tip_x + 10, tip_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Display frame info
        cv2.putText(frame, "Hand Tracking - Press 'q' to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Hand Tracking", frame)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Clean up
    color_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()
    hands.close()
