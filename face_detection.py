import cv2

# 使用OpenCV的Haar級聯分類器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 讀取圖像
image = cv2.imread('path_to_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 檢測人臉
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 繪製檢測到的人臉
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
