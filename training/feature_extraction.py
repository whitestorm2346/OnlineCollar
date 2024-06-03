import tensorflow as tf
from keras_facenet import FaceNet
import numpy as np
import cv2

# 初始化 FaceNet 模型
embedder = FaceNet()

# 自定義數據生成器（如果有需要微調的話）
def data_generator(image_folder, batch_size):
    # 實現一個生成器，用於從文件中加載數據
    # 這裡僅為示例，具體實現需根據數據集格式進行調整
    while True:
        images = []
        for i in range(batch_size):
            # 加載圖像
            image_path = ...  # 獲取圖像路徑
            image = cv2.imread(image_path)
            image = cv2.resize(image, (160, 160))  # 假設 FaceNet 需要160x160的輸入
            images.append(image)
        yield np.array(images)

# 假設你有一個圖像文件夾和標註文件（如果需要微調）
train_gen = data_generator('path_to_images', batch_size=32)

# 如果需要微調模型，這裡展示了如何進行訓練
# FaceNet 是一個已經訓練好的模型，通常不需要重新訓練
# 以下代碼僅供參考，如果需要進行微調

# 訓練模型（這裡只是示例，通常不需要重新訓練 FaceNet）
# embedder.model.fit(train_gen, steps_per_epoch=100, epochs=10)

# 保存模型
# 由於 keras_facenet 封裝的模型不直接支持 .save 方法，我們保存模型權重
embedder.model.save_weights('../models/facenet_weights.h5')

# 如果需要在別處加載模型
# embedder.model.load_weights('facenet_weights.h5')

# 這裡是一個簡單的測試示例，用來展示如何提取面部嵌入向量
test_image = cv2.imread('path_to_test_image.jpg')
test_image = cv2.resize(test_image, (160, 160))
test_image = np.expand_dims(test_image, axis=0)

# 提取嵌入向量
embedding = embedder.embeddings(test_image)[0]
print(embedding)
