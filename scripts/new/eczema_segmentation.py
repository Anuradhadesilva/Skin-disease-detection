import os
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras import layers, models

def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    return img / 255.0

def load_data(image_dir, mask_dir, label_csv):
    df = pd.read_csv(label_csv)
    X, Y = [], []
    for _, row in df.iterrows():
        img = preprocess_image(os.path.join(image_dir, row['filename']))
        mask = preprocess_image(os.path.join(mask_dir, row['filename']))
        mask = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        mask = (mask > 127).astype(np.float32)
        X.append(img)
        Y.append(mask[..., np.newaxis])
    return np.array(X), np.array(Y)

def build_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)

    u1 = layers.UpSampling2D()(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X, Y = load_data("data/images_flat", "data/masks", "data/labels.csv")
    model = build_unet()
    model.fit(X, Y, epochs=10, batch_size=8, validation_split=0.1)
    os.makedirs("models", exist_ok=True)
    model.save("models/eczema_unet.h5")
