import cv2 as cv

img = cv.imread(r"path_to_image")
# cv.imshow("Face of one person", img)

# greyscaling the image 
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("The grayed face", gray)

# haarcascade 
haar_cascade = cv.CascadeClassifier(r"haarfaces.xml_path")

# rectangle on faces 
faces_rectangle = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
print(f"Number of faces found: {len(faces_rectangle)}")

for (x,y,w,h) in faces_rectangle:
    cv.rectangle(gray, (x,y), (x+w, y+h), (0,198,211), thickness=3)
cv.imshow("Detected faces", gray)

cv.waitKey(0)