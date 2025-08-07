import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Load models
unet_model_path = r"/Users/desilvaanuradha/Documents/FYP/Ezcema-Detection/models/eczema_unet.h5"
classifier_model_path = r"/Users/desilvaanuradha/Documents/FYP/Ezcema-Detection/models/eczema_classifier.h5"

if not os.path.exists(unet_model_path) or not os.path.exists(classifier_model_path):
    raise FileNotFoundError("One or both models not found!")

unet_model = tf.keras.models.load_model(unet_model_path)
classifier_model = tf.keras.models.load_model(classifier_model_path)

def preprocess_image(img_path, target_size=(256, 256)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img_norm = img / 255.0
    return img_norm, img

def predict_and_segment(img_path):
    img_norm, original_img = preprocess_image(img_path)
    input_img = np.expand_dims(img_norm, axis=0)

    # Predict segmentation mask (values between 0 and 1)
    mask_pred = unet_model.predict(input_img)[0, :, :, 0]

    # Threshold mask to binary
    mask_binary = (mask_pred > 0.5).astype(np.uint8) * 255

    # Find contours on mask
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original image
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow boxes

    # Classify masked image (apply mask)
    masked_img = img_norm * (mask_binary[..., None] / 255.0)
    masked_img_batch = np.expand_dims(masked_img, axis=0)
    class_pred = classifier_model.predict(masked_img_batch)[0][0]
    label = "Eczema" if class_pred > 0.5 else "No Eczema"
    confidence = class_pred if class_pred > 0.5 else 1 - class_pred

    # Show results
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(original_img_rgb)
    plt.title(f"{label} - Confidence: {confidence:.2f}")
    plt.axis("off")
    plt.show()

    return label, confidence

# Usage example
img_path = r"/Users/desilvaanuradha/Documents/FYP/Ezcema-Detection/test_images/img_14.jpg"
predict_and_segment(img_path)
