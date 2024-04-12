import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui


def calulate_mining_angle(shoulder, elbow, wrist):
    shoulder = np.array(wrist)
    elbow = np.array(elbow)
    wrist = np.array(shoulder)

    radians = np.arctan2(shoulder[1]-elbow[1], shoulder[0]-elbow[1]) - np.arctan2(wrist[1]-elbow[1], wrist[0]-elbow[0])
    angle = np.abs(radians * 180 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle

    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Access the webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

screen_width, screen_height = pyautogui.size()
wrist_above_shoulder = False
mining_state = False
arm_closed = False
# last_arm_change_time = time.time()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Loop to capture frames from the webcam
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Flip the frame horizontaly
        frame = cv2.flip(frame ,1)

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
            
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            nose_x = int(nose.x * screen_width)
            nose_y = int(nose.y * screen_height)

            wristR = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y] 
            elbowR = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y] 
            shoulderR = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y] 

            angle = calulate_mining_angle(shoulderR, elbowR, wristR)

            # if angle < 50:  # Arm is considered closed
            #     if not arm_open:  # Start timer only when arm transitions from closed to open
            #         arm_open = True
            #         last_arm_change_time = time.time()
            # else:
            #     arm_open = False

            # if arm_open:
            #     if not mining_state:
            #         if time.time() - last_arm_change_time <= 3:  # Within 3 seconds to transition
            #             mining_state = True
            #             print("Mining started")
            #     elif time.time() - last_arm_change_time > 3:  # After 6 seconds of inactivity, mining stops
            #         mining_state = False
            #         print("Mining stopped due to inactivity")

            if angle < 50:
                if arm_closed == False:
                    arm_closed = True
                    mining_state = True
                    change_state = time.time()
                    print("TEST")
            else:
                arm_closed = False

            if arm_closed:
                if mining_state:
                    # Something is wrong below here
                    if change_state >= 6:  # If the change state is the same after 3 second then mining is not continued
                        print(change_state)
                        mining_state = False
                        print("Mining stopped")
            else:
                pass


            if mining_state:
                # Add code to perform mining action here
                print("Mining action")

            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbowR, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                       
            if(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y > landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y):
                wrist_above_shoulder = True
                if wrist_above_shoulder:
                    print("W")
            else:
                wrist_above_shoulder = False
                # print("S")

            pyautogui.moveTo(nose_x, nose_y)
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
