#Real time face detection: detection on video or camera using OPENCV
import cv2
from random import randrange
#making a detector/classifier 
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#choose the webcam to detect faces
webcam=cv2.VideoCapture(0) #for video-> cv2.VideoCapture(<file-path>)
while True:
    successful_frame_read, frame=webcam.read()
    grayscaled_vid=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_vid)
    for (x,y,w,h) in face_coordinates:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(255),randrange(255),randrange(255)),5)
    cv2.imshow('face detector',frame)
    key=cv2.waitKey(1)
    #stop if Q key is pressed
    if key==81 or key==113:
        break
webcam.release()


