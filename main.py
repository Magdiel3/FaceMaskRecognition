import cv2
import os

# Download the face recognition XML config file
cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

# Initialize algorithm
faceCascade = cv2.CascadeClassifier(cascPath)

# Select the integrated webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the image to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect the faces in the image
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,           # Parameter specifying how much the image size is reduced at each image scale. [1.05]
                                         minNeighbors=5,            # Parameter specifying how many neighbours each candidate rectangle should have to retain it. [3~6]
                                         minSize=(60, 60),          # Minimum possible object size. Objects smaller than that are ignored.
                                         flags=cv2.CASCADE_SCALE_IMAGE) # Mode of operation

    # Draw a rectangle around every detected face and add label
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
        cv2.putText(frame, 'FACE', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)

    # Display image
    cv2.imshow('Video', frame)

    # Define the quit key (q)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()