import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true), axis=None)

# Load the pre-trained expression model with any custom objects
expression_model = load_model(
    "C:\\Users\\girli\\EC_Utbildning\\Deep_Learning_Kunskapskontroll_2\\Final\\expression_model.keras",
    custom_objects={'custom_mse': custom_mse}
)

# Load the face detection model
face_cascade = cv2.CascadeClassifier(
    "C:\\Users\\girli\\EC_Utbildning\\Deep_Learning_Kunskapskontroll_2\\Final\\haarcascade_frontalface_default.xml"
)

# Define expression labels
expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_gray, (48, 48))
        face_array = img_to_array(face_resized)
        face_array = np.expand_dims(face_array, axis=0)
        face_array /= 255.0

        # Predict expression
        expression_pred = expression_model.predict(face_array)
        expression_label = expression_labels[np.argmax(expression_pred)]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'Expression: {expression_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Face Expression Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
