import os
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2

def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    return img / 255.0

def load_data(image_dir, mask_dir, label_csv):
    df = pd.read_csv(label_csv)
    X, Y, labels = [], [], []
    for _, row in df.iterrows():
        img = preprocess_image(os.path.join(image_dir, row['filename']))
        mask = preprocess_image(os.path.join(mask_dir, row['filename']))
        mask = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        mask = (mask > 127).astype(np.float32)
        X.append(img)
        Y.append(mask[..., np.newaxis])
        labels.append(row['label'])
    return np.array(X), np.array(Y), np.array(labels)

def build_classifier(input_shape=(256, 256, 3)):
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X, masks, labels = load_data("data/images_flat", "data/masks", "data/labels.csv")
    unet_model = load_model("models/eczema_unet.h5")
    predicted_masks = unet_model.predict(X)
    masked_images = X * predicted_masks

    X_train, X_test, y_train, y_test = train_test_split(masked_images, labels, test_size=0.2)
    model = build_classifier()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=8)
    os.makedirs("models", exist_ok=True)
    model.save("models/eczema_classifier.h5")
