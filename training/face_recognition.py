import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from keras_facenet import FaceNet
import joblib
import os

# 初始化 FaceNet 模型
embedder = FaceNet()

# 假設我們有訓練數據的文件夾和對應的標籤
image_folder = 'path_to_images'
labels_file = 'path_to_labels.txt'

# 加載圖像和標籤
def load_data(image_folder, labels_file):
    images = []
    labels = []
    with open(labels_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            image_path = os.path.join(image_folder, parts[0])
            label = parts[1]
            image = cv2.imread(image_path)
            image = cv2.resize(image, (160, 160))
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

# 提取嵌入向量
def extract_embeddings(images):
    embeddings = embedder.embeddings(images)
    return embeddings

# 加載數據
images, labels = load_data(image_folder, labels_file)

# 提取特徵
embeddings = extract_embeddings(images)

# 訓練 k-NN 模型
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(embeddings, labels)

# 保存 k-NN 模型
joblib.dump(knn, '../models/knn_face_recognition_model.pkl')

# 測試模型
def recognize_face(image_path, knn, embedder):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (160, 160))
    embedding = embedder.embeddings(np.array([image]))[0]
    label = knn.predict([embedding])
    return label

# 測試示例
test_image_path = 'path_to_test_image.jpg'
recognized_label = recognize_face(test_image_path, knn, embedder)
print(f'Recognized Label: {recognized_label[0]}')
