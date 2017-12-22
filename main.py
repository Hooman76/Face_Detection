import cv2
import sys

'''****** Get image from user ******'''
imagePath = sys.argv[1]

'''****** Creating the Haar Cascade for face detection ******'''
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

'''****** Reading and converting the given image ******'''
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converting colored scale image into grayed scale image

'''****** Detect the faces in the given image ******'''
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30)
)
print("Found {0} faces!".format(len(faces)))

'''****** Draw rectangles around the faces ******'''
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
