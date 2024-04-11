import cv2

# Access the webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Loop to capture frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    frame = cv2.flip(frame ,1)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Type 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close and destroy everything
cap.release()
cv2.destroyAllWindows()
