import json
import numpy as np
from PIL import Image, ImageDraw
import os

json_dir = "eczema_dataset/images"
output_base = "eczema_dataset/masks"
os.makedirs(output_base, exist_ok=True)

for file in os.listdir(json_dir):
    if file.endswith(".json"):
        json_path = os.path.join(json_dir, file)
        with open(json_path) as f:
            data = json.load(f)

        imgWidth = data['imageWidth']
        imgHeight = data['imageHeight']

        mask = Image.new('L', (imgWidth, imgHeight), 0)
        draw = ImageDraw.Draw(mask)

        for shape in data['shapes']:
            points = shape['points']
            polygon = [tuple(point) for point in points]
            draw.polygon(polygon, outline=1, fill=1)

        mask_np = np.array(mask) * 255  # make white mask (0=black, 255=white)
        name = file.replace('.json', '.png')
        mask.save(os.path.join(output_base, name))

        print(f"Saved mask: {name}")
