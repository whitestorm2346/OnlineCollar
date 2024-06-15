from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import cv2
import os

def create_pet_face_detection_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(4, activation='linear')  # 4 個輸出，分別是 x, y, width, height
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def load_data(image_folder, labels_file):
    images = []
    labels = []

    with open(labels_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            image_path = os.path.join(image_folder, parts[0])
            label = list(map(float, parts[1:5]))  # x, y, width, height

            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            image = image / 255.0  
            
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

image_folder = '../data/images'
labels_file = 'path_to_labels.txt'

X_train, y_train = load_data(image_folder, labels_file)

model = create_pet_face_detection_model()

model.fit(X_train, y_train, epochs=10, batch_size=32)
model.save('../models/face_detection_model.h5')

