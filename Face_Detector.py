import cv2
from random import randrange
#making a detector/classifier 
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#choose the image to detect faces in 
img=cv2.imread('TomHolland.jpg')
#convert the pic in to grayscale:
grayscaled_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#detect faces:
face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(255),randrange(255),randrange(255)),5)
#showing detected image on screen
cv2.imshow('face detector',img)
#pauses the excution of a code: wait until a key is pressed
cv2.waitKey()








