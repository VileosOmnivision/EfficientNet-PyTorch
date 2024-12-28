import cv2
import numpy as np
import os
import csv
from datetime import datetime
import re

def natural_sort_key(file_name):
    """
    Generate a key for natural sorting (handles numeric parts of filenames).
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file_name)]

def filter_by_hsv_channel(image, channel):
    """
    Filters the image, returning only pixels with HSV 'Value' above the 60th percentile of the left part.
    """
    # Compute the 60th percentile of the Value (V) channel
    v_channel = image[:, :, channel] 
    threshold = np.percentile(v_channel, 80)

    # Create a mask for pixels with V > threshold
    mask = image[:, :, 2] > threshold

    # Create a filtered image
    filtered_image = np.zeros_like(image)  # Black image with the same shape
    filtered_image[mask] = image[mask]  # Apply the mask

    # cv2.imshow("name",filtered_image)
    # cv2.waitKey(1)
    return filtered_image, mask

def preprocessing(image_path):
    """
    Transform an image into HSV.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def divider(hsv_image):
    """
    Divide the image into three vertical parts and get the average H, S, and V values
    for the middle part.
    """
    height, width, _ = hsv_image.shape
    middle_width = width // 2
    left_part = hsv_image[10:-40, 50:middle_width]
    masked_by_value = True

    if masked_by_value:
        filtered_img, mask = filter_by_hsv_channel(left_part, 2)
        filtered_img, mask = filter_by_hsv_channel(filtered_img, 1)

        # Mask the HSV channels
        h_channel = filtered_img[:, :, 0][mask]
        s_channel = filtered_img[:, :, 1][mask]
        v_channel = filtered_img[:, :, 2][mask]

        # Calculate the means
        avg_h = np.mean(h_channel) if h_channel.size > 0 else 0
        avg_s = np.mean(s_channel) if s_channel.size > 0 else 0
        avg_v = np.mean(v_channel) if v_channel.size > 0 else 0
        
        cv2.imshow("H",filtered_img[:, :, 0]*mask)
        cv2.waitKey(1)
        cv2.imshow("S",filtered_img[:, :, 1])
        cv2.waitKey(1)
        cv2.imshow("V",filtered_img[:, :, 2])
        cv2.waitKey(1)
    else:
        avg_h = np.mean(left_part[:, :, 0])
        avg_s = np.mean(left_part[:, :, 1])
        avg_v = np.mean(left_part[:, :, 2])
    
    return avg_h, avg_s, avg_v

def color_identifier(avg_h, avg_s):
    """
    Evaluate if the middle part has more green, yellow, or red based on H value.
    HSV Hues:
    - Red: 0-10 and 160-180
    - green: ~81-100
    - Yellow: ~26-35
    """
    if avg_h < 30 and avg_s > 180:
        return "red"
    else:
        return "green"

def process_images_in_folder(folder_path, output_csv_path):
    """
    Process all images in a folder and write the results to a CSV file.
    """
    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Datetime", "Image Name", "Average H", "Average S", "Average V", "Color"])
        
        # List files and sort them naturally
        file_list = sorted(
            [file_name for file_name in os.listdir(folder_path) if file_name.lower().endswith(('.png', '.jpg', '.jpeg'))],
            key=natural_sort_key
        )
        
        for file_name in file_list:
            image_path = os.path.join(folder_path, file_name)
            
            try:
                hsv_image = preprocessing(image_path)
                avg_h, avg_s, avg_v = divider(hsv_image)
                color = color_identifier(avg_h, avg_s)
                
                csv_writer.writerow([datetime.now(), file_name, avg_h, avg_s, avg_v, color])
                print(f"Processed: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

#cv2.namedWindow("name",cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("H",cv2.WINDOW_NORMAL)
cv2.namedWindow("S",cv2.WINDOW_NORMAL)
cv2.namedWindow("V",cv2.WINDOW_NORMAL)

# Example usage
folder_paths = [
    # "C:/chapus/fotorrojo/VideoRec_00013",
    "C:\chapus\fotorrojo\EXTRAIDO_vid-20241128_124624",
    # "C:/chapus/fotorrojo/VideoRec_00077",
    # "C:/chapus/fotorrojo/VideoRec_00097",
    # "C:/chapus/fotorrojo/VideoRec_00233",
]

for folder_path in folder_paths:
    output_csv_path = os.path.basename(folder_path).split(".")[0] + ".csv"
    process_images_in_folder(folder_path, output_csv_path)
