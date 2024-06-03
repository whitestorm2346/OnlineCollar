import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.api.preprocessing.image import ImageDataGenerator
import numpy as np

# 假設我們有一個自定義數據生成器來生成訓練數據和標籤
def data_generator(image_folder, annotation_file, batch_size):
    # 實現一個生成器，用於從文件中加載數據和標籤
    # 這裡僅為示例，具體實現需根據數據集格式進行調整
    while True:
        images = []
        labels = []
        # 加載圖像和標註
        for i in range(batch_size):
            # 加載圖像和對應的標註
            image = ""  # 加載圖像
            label = ""  # 加載標註，例如人臉的邊界框
            images.append(image)
            labels.append(label)
        yield np.array(images), np.array(labels)

# 定義模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4)  # 輸出4個值，表示邊界框的x, y, w, h
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# 構建數據生成器
train_gen = data_generator('path_to_images', 'path_to_annotations', batch_size=32)

# 訓練模型
model.fit(train_gen, steps_per_epoch=100, epochs=10)

# 保存模型
model.save('../models/face_detection_model.h5')

