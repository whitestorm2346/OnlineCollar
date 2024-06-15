import numpy as np
import cv2
import os
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split


def create_feature_extraction_model(input_shape=(224, 224, 3), embedding_size=128):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        GlobalAveragePooling2D(),  # 使用全局平均池化
        
        Dense(embedding_size, activation='relu')  # 最終輸出特徵向量
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def load_and_preprocess_images(image_folder, image_size=(224, 224)):
    images = []
    labels = []  # 這裡的 labels 是狗的不同品種或其他分類標籤

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)
        image = image / 255.0  
        images.append(image)
        
        label = int(image_name.split('_')[0].replace('dog', ''))
        labels.append(label)
    
    return np.array(images), np.array(labels)


image_folder = 'path_to_dog_images'

X, y = load_and_preprocess_images(image_folder)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_feature_extraction_model(embedding_size=128)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

model.save('../models/feature_extraction_model.h5')
