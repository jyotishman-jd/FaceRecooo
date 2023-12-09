import cv2
import os

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

id = input("enter id: ")
name = input("enter name: ")

# Create a directory to store images if it doesn't exist
output_folder = "dataSet"
os.makedirs(output_folder, exist_ok=True)

sampleNum = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        sampleNum += 1
        # Save the images in the "dataSet" folder with a specific format
        img_path = os.path.join(output_folder, f"User.{id}.{sampleNum}.jpg")
        cv2.imwrite(img_path, gray[y:y + h, x:x + w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.waitKey(1000)

    cv2.imshow("Face", img)
    cv2.waitKey(1)

    if sampleNum > 20:
        break

cam.release()
cv2.destroyAllWindows()
