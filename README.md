<img width="505" height="419" alt="Screenshot 2025-07-25 at 08 31 09" src="https://github.com/user-attachments/assets/4bf74ec0-8f23-4037-b548-0e5451406ee9" />🩺 AI-Based Skin Disease Detection System

This project implements a deep learning-based system to detect common skin diseases — **eczema, psoriasis, ringworm**, and more — from real-world, uncropped images. The system combines **U-Net** for lesion segmentation and **EfficientNet-B0** for classification. I have use dataset from various skin tones. The user can upload a real-world image without cropping.

---

## 📸 Example Results

Below are some sample predictions showing:
- Original image
- Segmented lesion
- Final predicted disease label

Orginal Image
> ![img_27](https://github.com/user-attachments/assets/70508e2f-bb0f-4e35-8796-f0e751198610)

Segmented lesion
> <img width="499" height="325" alt="Screenshot 2025-08-02 at 16 41 51" src="https://github.com/user-attachments/assets/ee7adb34-6221-4971-a70d-37a8a8d6b69f" />

Final predicted disease label
> <img width="1004" height="441" alt="Screenshot 2025-08-02 at 16 36 44" src="https://github.com/user-attachments/assets/e508d6a4-c159-4ae0-a6e6-246c9324dc15" />

Web UI
<img width="386" height="608" alt="Screenshot 2025-08-07 at 11 43 08" src="https://github.com/user-attachments/assets/644cfe8e-16e2-488b-ad80-47be79e307f3" />

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
```
### 📉 Confusion Matrix (EfficientNet-B0)

<img width="505" height="419" alt="Screenshot 2025-07-25 at 08 31 09" src="https://github.com/user-attachments/assets/fc444951-8c3b-4787-a884-30657ddab8d7" />


### 📈 Training Curves

<img width="306" height="132" alt="Screenshot 2025-08-02 at 16 44 01" src="https://github.com/user-attachments/assets/f70926e3-1684-4094-a2d7-ef46e84621d4" />




