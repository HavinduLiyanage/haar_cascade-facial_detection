import os 
import cv2 as cv
import numpy as np 

people = ["Names of the faces list"]
Dir = " " #dir to the images of faces file list 

haar_cascade = cv.CascadeClassifier(" ") # haarcascade.xml file path 

features = []
labels = [] # labeling the names of the faces 

def create_train():
    for person in people:
        path = os.path.join(Dir, person)
        label = people.index(person) # get the label of person 

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue # skip if the image can't be loaded 

            grey = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rectangle = haar_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=3)

            for (x,y,w,h) in faces_rectangle:
                faces_roi = grey[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print("Training is done")

# print(f"length of features :{len(features)}")
# print(f"length of labels :{len(labels)}")

# convering list into numpy array
features = np.array(features, dtype="object")
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train the recognizer with the features and label the list 
face_recognizer.train(features, labels)

#saving it on yml therefore it would be easier to use 
face_recognizer.save("faces_trained.yml")

# saving fatures and labels 
np.save('features.npy', features)
np.save('labels.npy', labels)

