![img_27](https://github.com/user-attachments/assets/90d3c848-dc6d-4b7d-a85e-fb9d7c321abf)![img_27](https://github.com/user-attachments/assets/db19ac0f-2708-48cb-9c25-d9c09d99413c) 🩺 AI-Based Skin Disease Detection System

This project implements a deep learning-based system to detect common skin diseases — **eczema, psoriasis, ringworm**, and more — from real-world, uncropped images. The system combines **U-Net** for lesion segmentation and **EfficientNet-B0** for classification.

---

## 📸 Example Results

Below are some sample predictions showing:
- Original image
- Segmented lesion
- Final predicted disease label

> _You can add output images here:_  
> `![Orginal](https://github.com/user-attachments/assets/fd72ffd5-6fd8-4507-a4b7-2a97066c2338)`  
> `![Uploading Screenshot 2025-08-02 at 16.41.51.png…]()` 
 
> `![Prediction](images/prediction_sample.jpg)`

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
