import cv2
import os
from ezcema_segmentation import segment_ezcema

def preprocess_and_save(image_path, save_path, img_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load: {image_path}")
        return
    image = segment_ezcema(image)

    image = cv2.resize(image, img_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if not save_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        save_path += ".jpg"

    cv2.imwrite(save_path, image)

def process_dataset(dataset_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for category in ["Acne", "No_Acne"]:
        input_folder = os.path.join(dataset_path, category)
        output_folder = os.path.join(output_path, category)
        os.makedirs(output_folder, exist_ok=True)

        for img_name in os.listdir(input_folder):
            input_path = os.path.join(input_folder, img_name)
            output_path_img = os.path.join(output_folder, img_name)
            preprocess_and_save(input_path, output_path_img)

    print("✅ Preprocessing & segmentation completed!")

if __name__ == "__main__":
    dataset_path = r"/Users/desilvaanuradha/Documents/FYP/Ezcema-Detection/Dataset/Train"
    output_path = r"/Users/desilvaanuradha/Documents/FYP/Ezcema-Detection/Dataset/Processed
    "
    process_dataset(dataset_path, output_path)

