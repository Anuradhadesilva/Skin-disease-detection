![img_27](https://github.com/user-attachments/assets/db19ac0f-2708-48cb-9c25-d9c09d99413c) 🩺 AI-Based Skin Disease Detection System

This project implements a deep learning-based system to detect common skin diseases — **eczema, psoriasis, ringworm**, and more — from real-world, uncropped images. The system combines **U-Net** for lesion segmentation and **EfficientNet-B0** for classification.

---

 📷 Example Results

> Add your image outputs below this section — like original image, segmentation mask, and predicted label.

| Original Image | Segmentation Mask | Predicted Class |
|----------------|-------------------|------------------|
| ![i<img width="499" height="325" alt="Screenshot 2025-08-02 at 16 41 51" src="https://github.com/user-attachments/assets/b5348c75-fb7e-4299-bf1d-c59ff67d85a7" />
mg_27](https://github.com/user-attachments/assets/8a411a0a-fefe-4fc0-af33-4861b7b6bb64)
 |  | Psorsis ✅ |

<img width="1004" height="441" alt="Screenshot 2025-08-02 at 16 36 44" src="https://github.com/user-attachments/assets/3adb8462-cb6c-4953-b7a1-f6f0875311c0" />

---

## 🚀 Features

- 🧠 **Two-stage architecture:** U-Net (segmentation) + EfficientNet-B0 (classification).
- 🧪 Handles **multiple classes**: eczema, psoriasis, ringworm, normal, others.
- 🔍 Supports both **connected and independent pipelines**.
- 🌍 Works well on **South Asian skin tones**, focusing on real-world usage.
- 🧑‍💻 Streamlit-based **web UI** for easy interaction.
- ⚕️ Built for **low-resource/rural areas** to assist early diagnosis.

Project Structure

```bash
project/
│
├── scripts/
│   ├── train_unet.py              # Train U-Net for lesion segmentation
│   ├── train_classifier.py        # Train EfficientNet classifier
│   ├── combine_predict.py         # Full pipeline (segmentation + classification)
│   ├── predict_classifier.py      # Classification only
│   ├── app.py                     # Streamlit web interface
│
├── models/
│   ├── unet_best.pth              # Trained U-Net model
│   ├── efficientnet_best.pth      # Trained EfficientNet-B0 model
│
├── dataset/
│   ├── images/                    # Input images
│   ├── masks/                     # Manual annotations (from LabelMe)
│   └── split/                     # train/val/test folders
│
├── requirements.txt
├── README.md
└── .gitignore
