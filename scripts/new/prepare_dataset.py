import os, shutil, pandas as pd, cv2
import numpy as np

eczema_dir = 'data/images/Eczema'
noeczema_dir = 'data/images/No_Eczema'
output_images = 'data/images_flat'
output_masks = 'data/masks'
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)

rows = []

def copy_images(src, label):
    for fname in os.listdir(src):
        if fname.lower().endswith((".jpg", ".png")):
            new_name = f"{label}_{fname}"
            shutil.copy(os.path.join(src, fname), os.path.join(output_images, new_name))
            rows.append({'filename': new_name, 'label': label})

            img = cv2.imread(os.path.join(src, fname))
            blank_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * (255 if label == 1 else 0)
            cv2.imwrite(os.path.join(output_masks, new_name), blank_mask)

copy_images(eczema_dir, 1)
copy_images(noeczema_dir, 0)

pd.DataFrame(rows).to_csv("data/labels.csv", index=False)
print("✅ Dataset prepared.")
