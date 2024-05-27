import face_recognition

# 加載已知人臉圖像並學習如何識別它們
known_image = face_recognition.load_image_file("known_image.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# 加載未知人臉圖像並提取特徵
unknown_image = face_recognition.load_image_file("unknown_image.jpg")
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# 比較未知人臉與已知人臉
results = face_recognition.compare_faces([known_encoding], unknown_encoding)

if results[0]:
    print("這是同一個人!")
else:
    print("這不是同一個人!")
