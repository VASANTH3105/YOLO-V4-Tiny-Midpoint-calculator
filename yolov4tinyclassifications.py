import cv2
import numpy as np
import darknet

# Load YOLOv4-tiny model and classes
net = darknet.load_net(b"yolov4-tiny.cfg", b"yolov4-tiny.weights", 0)
classes = ["class1", "class2", "class3", ...]

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

    # Display the detections on the frame
    for detection in detections:
        class_name = detection[0].decode()
        confidence = detection[1]
        x, y, w, h = detection[2]
        left = int(x - w/2)
        top = int(y - h/2)
        right = int(x + w/2)
        bottom = int(y + h/2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

'''
video capture using OpenCV, and in the main loop we read a frame from the video stream, c
onvert it to a Darknet image, and run object detection using the darknet.detect_image function. 

The detections variable contains a list of all the detected objects, where each object is represented by a tuple containing the class name, 
confidence score, and bounding box coordinates.

We then loop over the detections and draw a bounding box and label on the original frame using OpenCV's cv2.rectangle and cv2.putText functions. F
inally, we display the frame and wait for a key press to exit.

Note that you will need to replace the class names in the classes list with your own class names, 
and modify the paths to the YOLOv4-tiny configuration and weights files to match your own file locations. 
Also, make sure that you have the Darknet library installed on your system, and that you have the darknet.py Python wrapper file in your working directory.
'''
