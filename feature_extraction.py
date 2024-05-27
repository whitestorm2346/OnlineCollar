from keras_facenet import FaceNet
import cv2

# 初始化FaceNet模型
embedder = FaceNet()

# 加載和預處理人臉圖像
image = cv2.imread('path_to_face_image.jpg')
image = cv2.resize(image, (160, 160))  # 假設FaceNet需要160x160的輸入
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 提取嵌入向量
embedding = embedder.embeddings([image])[0]

print(embedding)  # 輸出128維的特徵向量
