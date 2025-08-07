 🩺 AI-Based Skin Disease Detection System

This project implements a deep learning-based system to detect common skin diseases — **eczema, psoriasis, ringworm**, and more — from real-world, uncropped images. The system combines **U-Net** for lesion segmentation and **EfficientNet-B0** for classification.

---

 📷 Example Results

> Add your image outputs below this section — like original image, segmentation mask, and predicted label.

| Original Image | Segmentation Mask | Predicted Class |
|----------------|-------------------|------------------|
| ![original](images/original.jpg) | ![mask](images/mask.jpg) | Eczema ✅ |

<img width="1004" height="441" alt="Screenshot 2025-08-02 at 16 36 44" src="https://github.com/user-attachments/assets/3adb8462-cb6c-4953-b7a1-f6f0875311c0" />

---

## 🚀 Features

- 🧠 **Two-stage architecture:** U-Net (segmentation) + EfficientNet-B0 (classification).
- 🧪 Handles **multiple classes**: eczema, psoriasis, ringworm, normal, others.
- 🔍 Supports both **connected and independent pipelines**.
- 🌍 Works well on **South Asian skin tones**, focusing on real-world usage.
- 🧑‍💻 Streamlit-based **web UI** for easy interaction.
- ⚕️ Built for **low-resource/rural areas** to assist early diagnosis.
