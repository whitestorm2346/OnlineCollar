from keras.src.models import load_model
from keras_facenet import FaceNet
import numpy as np
import joblib
import cv2

FACE_DETECTION_MODEL_PATH = 'models/face_detection_model.h5'
FEATURE_EXTRACTION_WEIGHTS_PATH = 'models/facenet_weights.h5'
FACE_RECOGNITION_MODEL_PATH = 'models/knn_face_recognition_model.pkl'

class FaceDetection:
    def __init__(self, model_path) -> None:
        self.model = load_model(model_path)

    def load_image(self, image_path):
        self.image = cv2.imread(image_path)

    def preprocess_image(self):
        image_resized = cv2.resize(self.image, (224, 224))
        image_normalized = image_resized / 255.0
        
        return np.expand_dims(image_normalized, axis=0)

    def crop_face(self) -> cv2.MatLike:
        prediction = self.model.predict(self.preprocess_image())[0]
    
        x, y, width, height = prediction
        rows, cols, _ = self.image.shape

        x = int(x * cols)
        y = int(y * rows)

        width = int(width * cols)
        height = int(height * rows)
        
        cropped_pet_face = self.image[y : y + height, x : x + width]

        return cropped_pet_face


class FeatureExtraction:
    def __init__(self, weights_path) -> None:
        self.model = FaceNet()
        self.model.load_weights(weights_path)

    def load_face(self, face_imgae):
        self.face = face_imgae

    def face_embedding(self):
        embedded_vector = ...

        return embedded_vector


class FaceRecognition:
    def __init__(self, model_path) -> None:
        self.model = joblib.load(model_path)

    def load_embedded_vector(self, embedded_vector):
        self.embedded_vector = embedded_vector

    def recognizing(self):
        match_result = ...

        return match_result


def main():
    face_detection_model = FaceDetection(FACE_DETECTION_MODEL_PATH)
    feature_extraction_model = FeatureExtraction(FEATURE_EXTRACTION_WEIGHTS_PATH)
    face_recognition_model = FaceRecognition(FACE_RECOGNITION_MODEL_PATH)

    image_path = ''

    face_detection_model.load_image(image_path)
    cropped_face = face_detection_model.crop_face()

    feature_extraction_model.load_face(cropped_face)
    embedded_vector = feature_extraction_model.face_embedding()

    face_recognition_model.load_embedded_vector(embedded_vector)
    match_face = face_recognition_model.recognizing()

    print(match_face)

if __name__ == "__main__":
    main()