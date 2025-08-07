import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
from io import BytesIO

from combine_predict import predict_uploaded_image  # U-Net + EfficientNet
from face_predictor import predict_face_skin        # VGG16-based model

# Page configuration
st.set_page_config(page_title="AI Skin Analyzer", layout="centered")
st.title("🧠 Smart Skin Detection Assistant")


task = st.selectbox("What do you want to analyze?", ["-- Select --", "Skin Disease", "Face Skin Issue"])


if task != "-- Select --":
    uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=400)

        with st.spinner("Analyzing image..."):
            try:
                if task == "Skin Disease":
                    highlighted_img, label, conf = predict_uploaded_image(
                        uploaded_file,
                        unet_ckpt="models/unet_best.pth",
                        clf_ckpt="models/efficientnet_best.pth",
                        data_dir="dataset"
                    )
                    st.success(f"🔍 Prediction: **{label}**")
                    st.info(f"📊 Confidence: **{conf * 100:.2f}%**")
                    st.image(highlighted_img, caption="Detected Lesion Area", width=400)

                elif task == "Face Skin Issue":
                    image = Image.open(uploaded_file).convert("RGB")


                    st.subheader("✂️ Crop the Face Region (Optional)")
                    use_crop = st.checkbox("Enable Cropping", value=True)

                    if use_crop:
                        cropped_img = st_cropper(image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
                        st.image(cropped_img, caption="Cropped Region", width=400)
                        input_img = cropped_img
                    else:
                        input_img = image


                    img_bytes = BytesIO()
                    input_img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)

                    label, conf, segmented = predict_face_skin(img_bytes)

                    st.success(f"🧴 Face Skin Prediction: **{label}**")
                    st.info(f"📊 Confidence: **{conf * 100:.2f}%**")

                    if segmented is not None:
                        st.image(segmented, caption="🔬 Segmented Region", width=400)

                else:
                    st.warning("Unknown task selected.")

            except Exception as e:
                st.error(f"Error: {e}")


# scripts/app.py
# import streamlit as st
# from PIL import Image
# from io import BytesIO
# from combine_predict import predict_uploaded_image  # Combined U-Net + EfficientNet
# from face_predictor import predict_face_skin
# from streamlit_cropper import st_cropper



# # Optional: face_predictor for face skin issues
# # from face_predictor import predict_face_skin

# st.set_page_config(page_title="AI Skin Analyzer", layout="centered")
# st.title("🧠 Smart Skin Detection Assistant")

# # Step 1: User chooses task
# task = st.selectbox("What do you want to analyze?", ["-- Select --", "Skin Disease", "Face Skin Issue"])

# # Step 2: Upload image
# if task != "-- Select --":
#     uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

#         with st.spinner("Analyzing image..."):
#             try:
#                 if task == "Skin Disease":
#                     highlighted_img, label, conf = predict_uploaded_image(
#                         uploaded_file,
#                         unet_ckpt="models/unet_best.pth",
#                         clf_ckpt="models/efficientnet_best.pth",
#                         data_dir="dataset"
#                     )
#                     st.success(f"🔍 Prediction: **{label}**")
#                     st.info(f"📊 Confidence: **{conf * 100:.2f}%**")

#                     # Show highlighted result
#                     st.image(highlighted_img, caption="Detected Lesion Area", use_column_width=True)
                    

#                elif task == "Face Skin Issue":
#     image = Image.open(uploaded_file).convert("RGB")
#     st.subheader("🖼️ Crop the Region to Analyze")
#     cropped_img = st_cropper(image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)

#     st.image(cropped_img, caption="Cropped Region", use_column_width=True)

#     with st.spinner("Analyzing cropped face area..."):
#         label, conf, segmented = predict_face_skin(cropped_img)
#         st.success(f"🧴 Face Skin Prediction: **{label}**")
#         st.info(f"📊 Confidence: **{conf * 100:.2f}%**")
#         st.image(segmented, caption="Segmented Area", use_column_width=True)


#                 else:
#                     st.warning("Unknown task selected.")

#             except Exception as e:
#                 st.error(f"❌ Error: {e}")

# scripts/app.py
# import streamlit as st
# from predict import predict_from_uploaded_image  # For skin
# # from face_predictor import predict_face_skin     # For face

# st.set_page_config(page_title="AI Skin Analyzer", layout="centered")
# st.title("🧠 Smart Skin Detection Assistant")

# # --- Step 1: User chooses which model to use ---
# task = st.selectbox("What do you want to analyze?", ["-- Select --", "Skin Disease", "Face Skin Issue"])

# # --- Step 2: Upload image ---
# if task != "-- Select --":
#     uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

#         # --- Step 3: Run prediction ---
#         with st.spinner("Analyzing image..."):
#             try:
#                 if task == "Skin Disease":
#                     label, conf = predict_from_uploaded_image(uploaded_file)
#                 elif task == "Face Skin Issue":
#                     label, conf = predict_from_uploaded_image(uploaded_file)
#                 else:
#                     label, conf = "Unknown", 0.0

#                 st.success(f"🔍 Prediction: **{label}**")
#                 st.info(f"📊 Confidence: **{conf * 100:.2f}%**")

#             except Exception as e:
#                 st.error(f"❌ Error: {e}")
