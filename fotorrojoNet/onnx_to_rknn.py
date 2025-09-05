import os
import time
import sys
import argparse
from datetime import datetime
import glob
import cv2
import numpy as np

from rknn.api import RKNN

def softmax(x):
    """Apply softmax function to input array"""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)

def check_outputs(index, image_path):
    """Display softmax probabilities in a readable format"""
    print(f"index: {index}")
    if 'norojo' in image_path:
        if index[0] > index[1]:
            print("CORRECTO! No es rojo")
            return True
        else:
            print("Fallo! No es norojo")
            return False
    else:
        if index[0] < index[1]:
            print("CORRECTO! Es rojo")
            return True
        else:
            print("Fallo! Es norojo")
            return False

def get_jpg_images(folder_path):
    """Get list of JPG images in a folder and its subfolders (recursive)"""
    if not os.path.exists(folder_path):
        print(f"Error: Test images folder '{folder_path}' not found!")
        return []

    jpg_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']
    images = []

    # Walk through the directory tree recursively
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if file has a JPG extension
            if any(file.endswith(ext) for ext in jpg_extensions):
                # Get the full path of the image
                full_path = os.path.join(root, file)
                images.append(full_path)

    if not images:
        print(f"Warning: No JPG images found in '{folder_path}' or its subfolders")

    return sorted(images)

def test_rknn_model(rknn_file_path, test_images_folder):
    """Test RKNN model with images from a folder"""
    if not os.path.exists(rknn_file_path):
        print(f"Error: RKNN file '{rknn_file_path}' not found!")
        return False

    # Get list of test images
    images_in_folder = get_jpg_images(test_images_folder)
    if not images_in_folder:
        return False

    print(f"Found {len(images_in_folder)} images to test")

    try:
        # Load RKNN model
        print("Loading RKNN model for testing...")
        rknn = RKNN()
        ret = rknn.load_rknn(rknn_file_path)
        if ret != 0:
            print("Error loading RKNN model!")
            return False

        # Initialize runtime
        ret = rknn.init_runtime(target='rk3588')
        if ret != 0:
            print("Error initializing RKNN runtime!")
            return False

        print("Starting inference tests...")
        right_counter = 0

        for img_path in images_in_folder:
            # Load and preprocess image in memory
            image = cv2.imread(img_path)  # Replace with your own image source
            if image is None:
                print(f"Error loading image: {img_path}")
                continue

            # Resize to model input size
            image = cv2.resize(image, (225, 75))

            # Convert to float32 and normalize to [0,1] to match PyTorch ToTensor()
            image = image.astype(np.float32)
            image = image / 255.0

            # Add batch dimension
            image = np.expand_dims(image, axis=0)

            for i in range(1):
                start_time = time.perf_counter()
                outputs = rknn.inference(inputs=[image])
                end_time = time.perf_counter()
                inference_time = end_time - start_time
                print(f"Inference {i+1}: {outputs} ({inference_time*1000:.2f}ms) image {img_path}")
                correct = check_outputs(softmax(np.array(outputs[0][0])), img_path)
                if correct:
                    right_counter += 1

        print(f"Total correct predictions: {right_counter}/{len(images_in_folder)} ({right_counter/len(images_in_folder)*100:.2f}%)")

        return True

    except Exception as e:
        print(f"Error during testing: {e}")
        return False
    finally:
        rknn.release()

def convert_onnx_to_rknn(onnx_file_path, output_name=None, quantization=False, test=None):
    """
    Convert ONNX model to RKNN format

    Args:
        onnx_file_path (str): Path to the input ONNX file
        output_name (str): Optional custom output name for RKNN file
        quantization (bool): Whether to enable quantization
    """
    if not os.path.exists(onnx_file_path):
        print(f"Error: ONNX file '{onnx_file_path}' not found!")
        return False

    # Generate output filename
    if output_name:
        rknn_output = f"{output_name}.rknn"
    else:
        # Use input filename without extension
        base_name = os.path.splitext(os.path.basename(onnx_file_path))[0]
        rknn_output = f"{base_name}.rknn"

    print(f"Converting {onnx_file_path} to {rknn_output}")


    try:
        # TODO: mean y std values copiados sin mirar de una resnet18 sobre imagenet
        # Configure RKNN with normalization parameters

        if not os.path.exists(rknn_output):
            # Load ONNX model
            print("Loading ONNX model...")

            # Initialize RKNN
            rknn = RKNN()
            rknn.config(
                mean_values=[[0.0, 0.0, 0.0]],
                std_values=[[1.0, 1.0, 1.0]],
                target_platform='rk3588'
            )
            ret = rknn.load_onnx(model=onnx_file_path)

            if ret != 0:
                print("Error loading ONNX model!")
                return False

            # Build the model
            print("Building RKNN model...")
            ret = rknn.build(do_quantization=quantization)
            if ret != 0:
                print("Error building RKNN model!")
                return False

            # Export RKNN model
            print(f"Exporting to {rknn_output}...")
            ret = rknn.export_rknn(rknn_output)
            if ret != 0:
                print("Error exporting RKNN model!")
                return False
            print(f"Successfully converted to {rknn_output}")
            rknn.release()

        else:
            print(f"RKNN model '{rknn_output}' already exists. Skipping conversion.")

        if test is not None:
            test_rknn_model(rknn_output, test)

    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert ONNX model to RKNN format or test RKNN model')
    parser.add_argument('input_file', help='Path to the input ONNX file (for conversion) or RKNN file (for testing)')
    parser.add_argument('-o', '--output', help='Custom output name (without extension) for conversion')
    parser.add_argument('-q', '--quantization', action='store_true',
                       help='Enable quantization during conversion')
    parser.add_argument('-t', '--test', nargs='?', const='/home/ubuntu/fotorrojo_ia/test_images/basic/',
                       help='Test mode: run inference on RKNN model with JPG images from folder (default: /home/ubuntu/fotorrojo_ia/)')

    args = parser.parse_args()

    convert_onnx_to_rknn(
        onnx_file_path=args.input_file,
        output_name=args.output,
        quantization=args.quantization,
        test=args.test
    )

if __name__ == "__main__":
    main()