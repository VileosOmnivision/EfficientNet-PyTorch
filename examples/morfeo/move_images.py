import os
import shutil
from tqdm import tqdm


def preparar_carpetas(output_path, input_path):
    # Define and create necessary directories
    train_folder = os.path.join(output_path, "train")
    val_folder = os.path.join(output_path, "val")
    train_rojo = os.path.join(train_folder, "rojo")
    train_norojo = os.path.join(train_folder, "norojo")
    val_rojo = os.path.join(val_folder, "rojo")
    val_norojo = os.path.join(val_folder, "norojo")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(train_rojo, exist_ok=True)
    os.makedirs(train_norojo, exist_ok=True)
    os.makedirs(val_rojo, exist_ok=True)
    os.makedirs(val_norojo, exist_ok=True)

def count_files_by_class(folder_path):
    # Initialize counters for images
    rojo_image_count = 0
    norojo_image_count = 0

    # Iterate through each subfolder type A in the input path
    for videos_extr in os.listdir(folder_path):
        carpeta_video = os.path.join(folder_path, videos_extr)    
        if not os.path.isdir(carpeta_video):
            continue
        # Count images in rojo folder
        rojo_folder = os.path.join(carpeta_video, "rojo")
        if os.path.exists(rojo_folder):
            rojo_image_count += len(os.listdir(rojo_folder))
        
        # Count images in norojo folder
        norojo_folder = os.path.join(carpeta_video, "norojo")
        if os.path.exists(norojo_folder):
            norojo_image_count += len(os.listdir(norojo_folder))
    return rojo_image_count, norojo_image_count

def split_and_move_images(input_path, output_path, color, ratio=0.8):
    train_folder = os.path.join(output_path, "train", color)
    val_folder = os.path.join(output_path, "val", color)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    for videos_extr in tqdm(os.listdir(input_path), desc="Procesando videos de color " + color):
        carpeta_video = os.path.join(input_path, videos_extr)
        color_folder = os.path.join(carpeta_video, color)

        if not os.path.isdir(carpeta_video):
            print("ERROR: No es una carpeta")
            continue
        if not os.path.exists(color_folder):
            print(f"ERROR: No existe la carpeta {color}")
            continue

        images = [img for img in os.listdir(color_folder) if img.endswith('.jpg')]
        print(f"Procesando {len(images)} imágenes de {color} en {carpeta_video}")

        # Add progress bar
        for i, image in enumerate(images):
            source_path = os.path.join(color_folder, image)
            if i % (1 / ratio) < 1:
                # Example: for ratio=0.8, 80% of the images will be moved to train folder
                destination_folder = train_folder
            else:
                destination_folder = val_folder
            shutil.move(source_path, destination_folder)


if __name__ == "__main__":
    # Ask the user for input and output paths
    input_path = input("Ingrese la ruta de entrada: ")
    output_path = input("Ingrese la ruta de salida (deje vacío para usar 'training_images' junto a la carpeta de entrada): ")

    if not output_path:
        output_path = os.path.join(os.path.dirname(input_path), "training_images")
        os.makedirs(output_path, exist_ok=True)
        print(f"Ruta de salida no proporcionada. Usando '{output_path}' como ruta de salida.")

    ratio = input("Ingrese la proporción de imágenes para entrenamiento (deje vacío para 80%): ")

    preparar_carpetas(output_path, input_path)
    new_rojo, new_norojo = count_files_by_class(input_path)
    print(f"Nuevas imágenes de rojo: {new_rojo}")
    print(f"Nuevas imágenes de norojo: {new_norojo}")

    old_rojo, old_norojo = count_files_by_class(output_path)
    print(f"Viejas imágenes de rojo: {old_rojo}")
    print(f"Viejas imágenes de norojo: {old_norojo}")
    
    # Split data into 80% for training and 20% for validation
    split_and_move_images(input_path, output_path, "rojo")
    split_and_move_images(input_path, output_path, "norojo")
