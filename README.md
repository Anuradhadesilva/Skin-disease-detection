🩺 AI-Based Skin Disease Detection System

This project implements a deep learning-based system to detect common skin diseases — **eczema, psoriasis, ringworm**, and more — from real-world, uncropped images. The system combines **U-Net** for lesion segmentation and **EfficientNet-B0** for classification.

---

## 📸 Example Results

Below are some sample predictions showing:
- Original image
- Segmented lesion
- Final predicted disease label

> _You can add output images here:_  
> [Original image]![img_27](https://github.com/user-attachments/assets/70508e2f-bb0f-4e35-8796-f0e751198610)

> <img width="499" height="325" alt="Screenshot 2025-08-02 at 16 41 51" src="https://github.com/user-attachments/assets/ee7adb34-6221-4971-a70d-37a8a8d6b69f" />
> <img width="1004" height="441" alt="Screenshot 2025-08-02 at 16 36 44" src="https://github.com/user-attachments/assets/e508d6a4-c159-4ae0-a6e6-246c9324dc15" />

 


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
