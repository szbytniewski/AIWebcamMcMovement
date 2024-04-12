import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Access the webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

wrist_above_shoulder = False

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Loop to capture frames from the webcam
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Flip the frame horizontaly
        # frame = cv2.flip(frame ,1)

        # Recolor to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect an pose 
        result = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = result.pose_landmarks.landmark
            if(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y > landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y):
                wrist_above_shoulder = True
                if wrist_above_shoulder:
                    print("W")
            else:
                wrist_above_shoulder = False
        except:
            pass

        # Render pose detection and set up colors of dots and lines
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(50, 0,255), thickness=2,circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(255,0,0), thickness=2,circle_radius=2))


        # Display the frame
        cv2.imshow('Webcam', image)

        # Type 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close and destroy everything
cap.release()
cv2.destroyAllWindows()
