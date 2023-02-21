import cv2
import numpy as np
import darknet

# Load YOLOv4-tiny model and classes
net = darknet.load_net(b"yolov4-tiny.cfg", b"yolov4-tiny.weights", 0)
classes = ["ball1", "ball2"]

# Set up video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to a Darknet image
    darknet_image = darknet.make_image(frame.shape[1], frame.shape[0], 3)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())

    # Run object detection on the Darknet image
    detections = darknet.detect_image(net, classes, darknet_image)

    # Find the two balls and calculate their midpoints
    ball1 = None
    ball2 = None
    for detection in detections:
        class_name = detection[0].decode()
        confidence = detection[1]
        x, y, w, h = detection[2]
        left = int(x - w/2)
        top = int(y - h/2)
        right = int(x + w/2)
        bottom = int(y + h/2)
        
        if class_name == "ball1":
            ball1 = (left + w/2, top + h/2)
        elif class_name == "ball2":
            ball2 = (left + w/2, top + h/2)
    
    if ball1 is not None and ball2 is not None:
        midpoint = ((ball1[0] + ball2[0])/2, (ball1[1] + ball2[1])/2)
        cv2.circle(frame, (int(midpoint[0]), int(midpoint[1])), 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()


