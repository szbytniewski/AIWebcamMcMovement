import time

import cv2
import mediapipe as mp
import numpy as np
import pydirectinput


def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

pydirectinput.PAUSE = 0

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Access the webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Constants
CURSOR_SPEED = 10

# Position variables
prev_right_shoulder_y = 0
prev_left_ankle_y, prev_right_ankle_y = 0, 0
first_nose_x, first_nose_y = 0,0

# Logic states variables
wrist_above_shoulder = False
mining_state = False
state = "down"
first_nose_detected = False
e_cliecked = False
left_clicked = False

# Other variables 
step_threshold = 0.005         
last_angle = 0
last_arm_change_time = time.time()
last_movment_instace = time.time()
motion_timeout = 1

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

        image_height, image_width, _ = frame.shape

        try:
            landmarks = result.pose_landmarks.landmark

            # jumping
            curr_right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y

            # mining
            wristR = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y] 
            elbowR = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y] 
            shoulderR = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y] 

            # walking
            ankleR = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
            ankleL = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y

            # movement
            nose_x = int(landmarks[mp_pose.PoseLandmark.NOSE.value].x * image_width)
            nose_y = int(landmarks[mp_pose.PoseLandmark.NOSE.value].y * image_height)

            # placing
            shoulderL = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            wristL = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y] 

            # left click right hold and open inventory (some of the points already called only added the ones that weren't)
            hipR = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            hipL = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y] 

            right_hand_in_trapezoid = (wristR[0] > shoulderR[0] and wristR[0] > hipR [0] and wristR[1] > shoulderR[1] and wristR[1] < hipR[1])
            left_hand_in_trapezoid = (wristL[0] < shoulderL[0] and wristL[0] < hipL [0] and wristL[1] > shoulderL[1] and wristL[1] < hipL[1])

            # opening inventory eating and single left clicking
            if (left_hand_in_trapezoid and right_hand_in_trapezoid):
                if not e_cliecked:
                    pydirectinput.press("e")
                    # print("open inventory")
                    e_cliecked = True
            elif (left_hand_in_trapezoid):
                pydirectinput.mouseDown(button='right')
                print("eating")
            elif (right_hand_in_trapezoid):
                if not left_clicked:
                    # doesn't work for some reason i don't know why
                    pydirectinput.mouseDown(button='left')
                    print("single click")
                    left_clicked = True
            else:
                pydirectinput.mouseUp(button='right')
                pydirectinput.mouseUp(button='left')
                if e_cliecked == True:
                    e_cliecked = False
                if left_clicked == True:
                    left_clicked = False

            # jumping condition
            if (curr_right_shoulder_y > prev_right_shoulder_y + 0.04 ):
                pydirectinput.keyDown("space")
                print("jump")
            else:
                pydirectinput.keyUp("space")
                pass
 
            # placing blocks condition
            prev_right_shoulder_y = curr_right_shoulder_y
            if(shoulderL[1] > wristL[1]):
                wrist_above_shoulder = True
                if wrist_above_shoulder:
                    pydirectinput.rightClick()
                    print("placing blocks")
                    # pass
            else:
                wrist_above_shoulder = False

            # mining condition
            angle = calculate_angle(shoulderR, elbowR, wristR)

            if angle < 30:
                if state != "up":
                    last_arm_change_time = time.time()  # Reset idle time if arm is in motion
                    state = "up"
                    mining_state = True
            if angle > 150 and state == "up":
                last_arm_change_time = time.time() # Reset idle time if arm is in motion
                state = "down"
                mining_state = True

            # Check if user is inactive 
            if time.time() - last_arm_change_time > motion_timeout:
                mining_state = False

            if mining_state:
                pydirectinput.mouseDown()
                print("Mining action")
                # pass
            else:
                pydirectinput.mouseUp()
                # pass


            # walking condition
            if ankleL > prev_right_ankle_y + step_threshold and ankleR < prev_left_ankle_y - step_threshold:
                last_movment_instace = time.time()
                pydirectinput.keyDown('ctrl')
                pydirectinput.keyDown('w')
                print("Step forward detected") 
                # pass

            if time.time() - last_movment_instace > motion_timeout:
                pydirectinput.keyUp('ctrl')
                pydirectinput.keyUp('w')
                # pass


            prev_right_ankle_y = ankleL
            prev_left_ankle_y = ankleR

            cv2.putText(image, str(angle), 
                tuple(np.multiply(elbowR, [640, 480]).astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )
            

            # movement condition
            if not first_nose_detected:
                first_nose_x = nose_x
                first_nose_y = nose_y
                first_nose_detected = True

            cursor_dx = 0
            cursor_dy = 0

            if nose_x < first_nose_x - 25:
                # Move left
                cursor_dx -= CURSOR_SPEED
                # print("LEFT")
            elif nose_x > first_nose_x + 25:
                # Move right
                cursor_dx += CURSOR_SPEED
                # print("RIGHT")

            if nose_y < first_nose_y - 25:
                # Move up
                cursor_dy -= CURSOR_SPEED
                # print("UP")
            elif nose_y > first_nose_y  + 25:
                # Move down
                cursor_dy += CURSOR_SPEED
                # print("DOWN")

            # Apply movement to cursor
            pydirectinput.moveRel(int(cursor_dx), int(cursor_dy)) 

        except:
            pass

        cv2.rectangle(image, (first_nose_x - 25, first_nose_y - 25), (first_nose_x + 25, first_nose_y + 25), (0, 255, 0), 2 )

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
 