import os
import pandas as pd
import shutil

# Function to move images based on color
def move_images(data, base_folder, subfolder):
    for index, row in data.iterrows():
        image_name = row['Image Name']
        color = row['Color']
        source_path = os.path.join(base_folder, image_name)
        destination_folder = os.path.join(base_folder, subfolder, color)
        
        # Create color directory if not exist
        os.makedirs(destination_folder, exist_ok=True)
        
        # Move the file
        shutil.copy(source_path, destination_folder)
        
# Define source and destination directories
source_folder = [
    "VideoRec_00075",
    "VideoRec_00077",
    ]
train_folder = "dataset/train"
val_folder = "dataset/val"

for folder in source_folder:
    # Load CSV
    csv_file = folder + ".csv"
    df = pd.read_csv(csv_file)

    # Split data into 80% for training and 20% for validation
    train_data = df.iloc[:int(len(df) * 0.8)]
    val_data = df.iloc[int(len(df) * 0.8):]

    # Create necessary directories if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Move the images to the respective folders
    move_images(train_data, folder, 'train')
    move_images(val_data, folder, 'val')
