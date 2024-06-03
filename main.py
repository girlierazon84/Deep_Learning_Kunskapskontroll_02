import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model # type: ignore
from mtcnn import MTCNN

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Load pre-trained models
gender_age_model = load_model('C:/Users/girli/EC_Utbildning/Deep_Learning_Kunskapskontroll_2/Final/gender_age_detection_model.keras')
expression_model = load_model('C:/Users/girli/EC_Utbildning/Deep_Learning_Kunskapskontroll_2/Final/model_expression.keras')

# Initialize MTCNN for face detection
detector = MTCNN()

# Mapping dictionary for gender and expression
GENDER_DICT = {0: 'Male', 1: 'Female'}
EXPRESSION_DICT = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

def preprocess_image_for_age_and_gender(face_img):
    face_img_resized = cv2.resize(face_img, (128, 128))
    face_img_normalized = face_img_resized / 255.0
    face_img_expanded = np.expand_dims(face_img_normalized, axis=0)
    return face_img_expanded

def predict_gender_and_age(face_img_expanded):
    gender_prediction, age_prediction = gender_age_model.predict(face_img_expanded)
    gender = GENDER_DICT[np.argmax(gender_prediction)]
    age = int(tf.keras.backend.eval(age_prediction))
    return gender, age

def preprocess_image_for_expression(face_img):
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    face_gray_resized = cv2.resize(face_gray, (48, 48))
    face_gray_normalized = face_gray_resized / 255.0
    face_gray_expanded = np.expand_dims(face_gray_normalized, axis=(0, -1))
    return face_gray_expanded

def predict_expression(face_gray_expanded):
    expression_prediction = expression_model.predict(face_gray_expanded)
    expression = EXPRESSION_DICT[np.argmax(expression_prediction)]
    return expression

def draw_bounding_box_and_labels(frame, x, y, width, height, gender, age, expression):
    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
    cv2.putText(frame, f'Gender: {gender}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f'Age: {age}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f'Expression: {expression}', (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Load the face detection cascade classifier
face_cascade_path = 'C:/Users/girli/EC_Utbildning/Deep_Learning_Kunskapskontroll_2/Final/haarcascade_frontalface_default.xml'
face_classifier = cv2.CascadeClassifier(face_cascade_path)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, width, height = face['box']
        face_img = rgb_frame[y:y+height, x:x+width]

        # Preprocess for age and gender model
        face_img_expanded = preprocess_image_for_age_and_gender(face_img)

        # Predict gender and age
        gender, age = predict_gender_and_age(face_img_expanded)

        # Preprocess for expression model
        face_gray_expanded = preprocess_image_for_expression(face_img)

        # Predict expression
        expression = predict_expression(face_gray_expanded)

        # Draw bounding box and labels
        draw_bounding_box_and_labels(frame, x, y, width, height, gender, age, expression)

    # Display the resulting frame
    cv2.imshow('Real-time Face Analysis', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
