import os

def rename_images_in_folder(folder_path, prefix="no_eczema"):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

    images.sort()  # Keep files in alphabetical order

    for i, filename in enumerate(images, start=1):
        ext = os.path.splitext(filename)[1]
        new_name = f"{prefix}_{i:02d}{ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        print(f"Renamed: {filename} -> {new_name}")

# CHANGE THIS TO YOUR FOLDER PATH
rename_images_in_folder(r"/Users/desilvaanuradha/Documents/FYP/Ezcema-Detection/data/images/No_Eczema")
