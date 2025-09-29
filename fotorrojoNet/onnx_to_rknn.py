import os
import time
import sys
import argparse
from datetime import datetime
import glob
import cv2
import numpy as np
import re

from rknn.api import RKNN
import random

def softmax(x):
    """Apply softmax function to input array"""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)

def extract_image_size_from_filename(filename):
    """
    Extract image dimensions from model filename.
    Expected format: modelname_HEIGHTxWIDTH.extension
    Examples:
    - fotorrojoNet_75x225.onnx -> (225, 75)
    - fotorrojoNet_20250905_1215_bilbao2_75x225.onnx -> (225, 75)

    Returns:
        tuple: (width, height) if found, None if not found
    """
    # Pattern to match dimensions in format: HEIGHTxWIDTH at the end before extension
    # This captures digits_x_digits pattern
    pattern = r'_(\d+)x(\d+)(?:\.[^.]+)?$'

    match = re.search(pattern, filename)
    if match:
        height = int(match.group(1))
        width = int(match.group(2))
        return (width, height)  # cv2.resize expects (width, height)

    return None

def check_outputs(index, image_path):
    """Display softmax probabilities in a readable format"""
    print(f"index: {index} -- path: {image_path} -- bool: {'norojo' in image_path}")
    if 'norojo' in image_path:
        if index[0] > index[1]:
            print("Fallo! No es rojo")
            return True
        else:
            print("CORRECTO! Es norojo")
            return False
    else:
        if index[0] < index[1]:
            print("CORRECTO! Es norojo")
            return True
        else:
            print("Fallo! No es rojo")
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

    # Extract image size from model filename
    model_filename = os.path.basename(rknn_file_path)
    extracted_size = extract_image_size_from_filename(model_filename)

    if extracted_size:
        target_height, target_width = extracted_size
        print(f"Extracted model input size from filename: {target_width}x{target_height}")
    else:
        print(f"Could not extract image size from filename: {model_filename}")
        print("Please provide the expected input size for the model.")
        try:
            size_input = input("Enter image size as WIDTHxHEIGHT (e.g., 225x75): ")
            width_str, height_str = size_input.split('x')
            target_width, target_height = int(width_str), int(height_str)
            print(f"Using provided size: {target_width}x{target_height}")
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or operation cancelled. Using default size 225x75")
            target_width, target_height = 75, 225

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
        near_misses = 0  # For predictions that were almost correct
        total_images = len(images_in_folder)
        wrong_examples = []
        inference_times = []

        for img_path in images_in_folder:
            # Load and preprocess image in memory
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error loading image: {img_path}")
                continue

            # Get original image dimensions
            original_height, original_width = image.shape[:2]
            original_size = (original_width, original_height)
            target_size = (target_width, target_height)

            # Check if resize is needed and print warning
            # if original_size != target_size:
            #     print(f"WARNING: Resizing image {os.path.basename(img_path)} from {original_width}x{original_height} to {target_width}x{target_height}")

            # Resize to model input size using BICUBIC interpolation to match training
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

            # Convert to float32 and normalize to [0,1] to match PyTorch ToTensor()
            image = image.astype(np.float32)
            image = image / 255.0

            # Add batch dimension to get the RKNN NHWC format: batch, height, width, channels
            image = np.expand_dims(image, axis=0)

            # Run inference
            start_time = time.perf_counter()
            outputs = rknn.inference(inputs=[image])
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            inference_times.append(inference_time)

            # Get detailed prediction analysis
            pred_scores = softmax(np.array(outputs[0][0]))
            pred_class = np.argmax(pred_scores)
            confidence = pred_scores[pred_class]

            # Determine expected class based on filename
            expected_class = 1 if 'rojo' in img_path and 'norojo' not in img_path else 0
            is_correct = (pred_class == expected_class)

            if is_correct:
                right_counter += 1
                result = "CORRECT"
            else:
                # Check if it was a near miss (confidence was close)
                if len(pred_scores) >= 2 and abs(pred_scores[0] - pred_scores[1]) < 0.2:
                    near_misses += 1
                    result = "NEAR MISS"
                else:
                    result = "WRONG"
                    wrong_examples.append((img_path, pred_class, confidence))

            # print(f"{result}: {os.path.basename(img_path)} - Predicted: {pred_class} ({confidence:.3f}), Expected: {expected_class}, Scores: {pred_scores} ({inference_time*1000:.2f}ms)")

        wrong_pred_folder = os.path.join(test_images_folder, '../wrong_predictions')
        print(f"\nSaving some wrong predictions in 'wrong_predictions' folder...")
        random.shuffle(wrong_examples)
        number_examples = min(20, len(wrong_examples))
        for wrong_img, pred_cls, conf in wrong_examples[:number_examples]:
            if not os.path.exists(wrong_pred_folder):
                os.makedirs(wrong_pred_folder)
            base_name = os.path.basename(wrong_img)
            save_path = os.path.join(wrong_pred_folder, f"pred{pred_cls}_conf{int(conf*100):02d}_{base_name}")
            image = cv2.imwrite(save_path, cv2.imread(wrong_img))

        # Calculate statistics
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        std_inference_time = np.std(inference_times) * 1000

        print(f"\n" + "="*60)
        print(f"RKNN MODEL RESULTS SUMMARY:")
        print(f"- Model: {os.path.basename(rknn_file_path)}")
        print(f"- Total images: {total_images}")
        print(f"- Correct predictions: {right_counter}")
        print(f"- Accuracy: {right_counter/total_images*100:.2f}%")
        print(f"- Near misses: {near_misses} ({near_misses/total_images*100:.2f}%)")
        print(f"- Combined (correct + near miss): {(right_counter + near_misses)/total_images*100:.2f}%")
        print(f"- Average inference time: {avg_inference_time:.2f} ± {std_inference_time:.2f} ms")
        print(f"="*60)

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

            # Initialize RKNN with improved configuration for better accuracy
            rknn = RKNN()
            rknn.config(
                mean_values=[[0.0, 0.0, 0.0]],
                std_values=[[1.0, 1.0, 1.0]],
                target_platform='rk3588',
                optimization_level=1,  # Lower optimization for better accuracy
                quantized_algorithm='normal',  # Specify algorithm explicitly
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

    # Just testing an already created model.
    # Usage (with venv_rknn active): 'python3 onnx_to_rknn.py mymodel.rknn -t'
    # test_rknn_model(args.input_file, args.test)

    convert_onnx_to_rknn(
        onnx_file_path=args.input_file,
        output_name=args.output,
        quantization=args.quantization,
        test=args.test
    )

if __name__ == "__main__":
    main()

