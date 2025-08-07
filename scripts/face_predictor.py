import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained Keras model
model = tf.keras.models.load_model("models/vgg16_acne_model.h5")

# Class labels
class_labels = {
    0: "Acne Detected",
    1: "Blackheads Detected",
    2: "Darkspots Detected",
    3: "Other"
}

# 🔴 Acne Area Segmentation
def segment_acne_cv2(image_array):
    hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 30:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image_array

# 🔍 Run Classification + Segmentation
def predict_face_skin(image_file):
    # Load and preprocess image
    image = Image.open(image_file).convert("RGB")
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # Predict class
    prediction = model.predict(img_input, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    label = class_labels.get(predicted_class, "Unknown")

    # Segment acne from original image
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    segmented_img = segment_acne_cv2(image_bgr)
    segmented_pil = Image.fromarray(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))

    return label, confidence, segmented_pil
