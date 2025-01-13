import os
from PIL import Image
import cv2
from tqdm import tqdm

def get_image_resolutions(root_folder):
    sizes = []
    for subdir, _, files in os.walk(root_folder):
        for file in tqdm(files, desc=f"Processing {subdir}"):
            if file.lower().endswith('.jpg'):
                file_path = os.path.join(subdir, file)
                image = cv2.imread(file_path)
                img_size = image.shape[:2]
                if img_size not in sizes:
                    sizes.append(img_size)
                    print(f"Image size: {img_size} - {file_path}")
                if image.shape[:2] != (25, 75):
                    image = pad_and_resize(image)
                if image.shape[:2][0] == 3 * image.shape[:2][1]:
                    image = cv2.hconcat([image, image, image])
                cv2.imwrite(file_path, image)


def pad_and_resize(image, target_size=(25, 75)):
    # Original dimensions
    h, w = image.shape[:2]
    target_ratio = 1 / 3  # width:height ratio

    # Calculate target height and width
    if w / h > target_ratio:
        new_h = int(w / target_ratio)
        new_w = w
    else:
        new_w = int(h * target_ratio)
        new_h = h

    # Calculate padding
    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left

    # Add padding
    padded_image = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # Resize to target dimensions
    resized_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_AREA)
    return resized_image

if __name__ == "__main__":
    root_folder = r"C:\datasets\dataset_fotorrojo"
    get_image_resolutions(root_folder)
