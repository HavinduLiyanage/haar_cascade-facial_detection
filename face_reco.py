import numpy as np
import cv2 as cv 

haar_cascade = cv.CascadeClassifier(r"Path to haarcascade.xml")

people = ["Names of the faces list"]

face_reco = cv.face.LBPHFaceRecognizer_create()
face_reco.read("faces_trained.yml")

img = cv.imread(r"path_to_face")

grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Person in grey", grey)

# detection of the face
face_rectangle = haar_cascade.detectMultiScale(grey, 1.1, 3)

for (x,y,w,h) in face_rectangle:
    faces_roi = grey[y:y+h,x:x+w]

    label, confidence = face_reco.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow("Detect Face", img)

cv.waitKey(0)