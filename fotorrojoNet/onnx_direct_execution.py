import os
import time
import sys
import argparse
from datetime import datetime
import glob
import cv2
import numpy as np
import re
import math
from collections import defaultdict
import onnxruntime as ort
import random

def softmax(x):
    """Apply softmax function to input array"""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)

def extract_metadata_from_filename(filename):
    """
    Extract location, day, month and day/night from filename.
    Expected format: LOCATION_YYYYMMDD_HHMMSS_frameXXX_semaforoX.jpg
    Examples:
    - 103_20250830_123015_frame293_semaforo0.jpg -> ubication=103, day=30, month=08, time_period='day'
    - 602_20250904_030015_frame227_semaforo1.jpg -> ubication=602, day=04, month=09, time_period='night'

    Returns:
        dict: {'ubication': int, 'day': int, 'month': int, 'time_period': str} or None if parsing fails
    """
    # Pattern to match: LOCATION_YYYYMMDD_HHMMSS_...
    pattern = r'^(\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_'

    match = re.match(pattern, filename)
    if not match:
        return None

    try:
        ubication = int(match.group(1))
        year = int(match.group(2))
        month = int(match.group(3))
        day = int(match.group(4))
        hour = int(match.group(5))
        minute = int(match.group(6))
        second = int(match.group(7))

        # Determine day/night based on hour (night: 22h-07h, day: 07h-22h)
        if hour >= 22 or hour < 7:
            time_period = 'night'
        else:
            time_period = 'day'

        return {
            'ubication': ubication,
            'day': day,
            'month': month,
            'time_period': time_period,
            'hour': hour
        }
    except (ValueError, IndexError):
        return None

def analyze_prediction_statistics(results_data):
    """
    Analyze prediction statistics by location, month, and time period.

    Args:
        results_data: List of tuples containing (filename, result_type, is_correct_or_near_miss)
                     where result_type is 'CORRECT', 'NEAR MISS', or 'WRONG'

    Returns:
        dict: Statistics organized by location-month pairs and time periods
    """
    # Initialize nested dictionaries for statistics
    stats = defaultdict(lambda: {
        'day': {'correct': 0, 'near_miss': 0, 'wrong': 0, 'total': 0},
        'night': {'correct': 0, 'near_miss': 0, 'wrong': 0, 'total': 0}
    })

    for filename, result_type, _ in results_data:
        metadata = extract_metadata_from_filename(filename)
        if not metadata:
            continue

        ubication = metadata['ubication']
        month = metadata['month']
        time_period = metadata['time_period']

        # Create location-month key
        location_month_key = f"Loc{ubication:03d}-M{month:02d}"

        # Update statistics
        stats[location_month_key][time_period]['total'] += 1

        if result_type == 'CORRECT':
            stats[location_month_key][time_period]['correct'] += 1
        elif result_type == 'NEAR MISS':
            stats[location_month_key][time_period]['near_miss'] += 1
        else:  # 'WRONG'
            stats[location_month_key][time_period]['wrong'] += 1

    return dict(stats)

def print_statistics_table(stats):
    """
    Print a formatted table with statistics.
    """
    if not stats:
        print("No statistics to display.")
        return

    # Calculate total images across all categories
    total_images = 0
    for location_month in stats:
        total_images += stats[location_month]['day']['total']
        total_images += stats[location_month]['night']['total']

    print(f"\n" + "=" * 70)
    print("PREDICTION STATISTICS BY LOCATION-MONTH AND TIME PERIOD")
    print("=" * 70)

    # Header
    header = f"{'Location-Month':<15} | {'DAY':<21} | {'NIGHT':<25}"
    print(header)
    print("-" * 70)

    subheader = f"{'':15} | {'Total':>6} {'Acc%':>6} {'Images%':>7} | {'Total':>6} {'Acc%':>6} {'Images%':>7}"
    print(subheader)
    print("-" * 70)

    # Sort by location-month key
    for location_month in sorted(stats.keys()):
        day_stats = stats[location_month]['day']
        night_stats = stats[location_month]['night']

        # Calculate percentages for day
        day_total = day_stats['total']
        if day_total > 0:
            day_acc = math.ceil((day_stats['correct'] / day_total) * 100)
            day_images_pct = math.ceil((day_total / total_images) * 100)
            day_total_str = f"{day_total:6d}"
            day_acc_str = f"{day_acc:6d}"
            day_images_str = f"{day_images_pct:6d}%"
        else:
            day_total_str = f"{'':5s}-"
            day_acc_str = f"{'':5s}-"
            day_images_str = f"{'':6s}-"

        # Calculate percentages for night
        night_total = night_stats['total']
        if night_total > 0:
            night_acc = math.ceil((night_stats['correct'] / night_total) * 100)
            night_images_pct = math.ceil((night_total / total_images) * 100)
            night_total_str = f"{night_total:6d}"
            night_acc_str = f"{night_acc:6d}"
            night_images_str = f"{night_images_pct:6d}%"
        else:
            night_total_str = f"{'':5s}-"
            night_acc_str = f"{'':5s}-"
            night_images_str = f"{'':6s}-"

        # Format row
        day_section = f"{day_total_str} {day_acc_str} {day_images_str}"
        night_section = f"{night_total_str} {night_acc_str} {night_images_str}"

        row = f"{location_month:<15} | {day_section} | {night_section}"
        print(row)

    print("="* 70)
    print(f"Total images analyzed: {total_images}")
    print("Legend: Total = Number of images, Acc% = Accuracy, Images% = Percentage of total dataset")
    print("="* 70)

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

def test_onnx_model(onnx_file_path, test_images_folder):
    """Test ONNX model with images from a folder using ONNX Runtime"""
    if not os.path.exists(onnx_file_path):
        print(f"Error: ONNX file '{onnx_file_path}' not found!")
        return False

    # Extract image size from model filename
    model_filename = os.path.basename(onnx_file_path)
    extracted_size = extract_image_size_from_filename(model_filename)

    if extracted_size:
        target_width, target_height = extracted_size
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
            target_width, target_height = 225, 75

    # Get list of test images
    images_in_folder = get_jpg_images(test_images_folder)
    if not images_in_folder:
        return False

    print(f"Found {len(images_in_folder)} images to test")

    try:
        # Load ONNX model with ONNX Runtime
        print("Loading ONNX model for testing...")

        # Create ONNX Runtime session with optimizations
        session = ort.InferenceSession(onnx_file_path, providers=['CPUExecutionProvider'])

        # Get input details
        input_details = session.get_inputs()[0]
        input_name = input_details.name
        input_shape = input_details.shape
        print(f"Model input: {input_name}, shape: {input_shape}")

        print("Starting inference tests...")
        right_counter = 0
        near_misses = 0  # For predictions that were almost correct
        total_images = len(images_in_folder)
        wrong_examples = []
        inference_times = []
        results_data = []  # Store results for statistics analysis

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

            # Resize to model input size
            image = cv2.resize(image, (target_width, target_height))

            # Convert BGR to RGB (OpenCV uses BGR, most models expect RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to float32 and normalize to [0,1] to match PyTorch ToTensor()
            image = image.astype(np.float32)
            image = image / 255.0

            # Convert to NCHW format (batch, channels, height, width) for PyTorch models
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
            image = np.expand_dims(image, axis=0)   # Add batch dimension

            # Run inference
            start_time = time.perf_counter()
            outputs = session.run(None, {input_name: image})
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
                is_correct_or_near_miss = True
            else:
                # Check if it was a near miss (confidence was close)
                if len(pred_scores) >= 2 and abs(pred_scores[0] - pred_scores[1]) < 0.2:
                    near_misses += 1
                    result = "NEAR MISS"
                    is_correct_or_near_miss = True
                else:
                    result = "WRONG"
                    is_correct_or_near_miss = False
                    wrong_examples.append((img_path, pred_class, confidence))

            # Store result data for statistics
            filename = os.path.basename(img_path)
            results_data.append((filename, result, is_correct_or_near_miss))

            print(f"{result}: {os.path.basename(img_path)} - Predicted: {pred_class} ({confidence:.3f}), Expected: {expected_class}, Scores: {pred_scores} ({inference_time*1000:.2f}ms)")

        # Save wrong predictions
        wrong_pred_folder = os.path.join(test_images_folder, '../wrong_predictions_onnx')
        print(f"\nSaving some wrong predictions in 'wrong_predictions_onnx' folder...")
        random.shuffle(wrong_examples)
        number_examples = min(20, len(wrong_examples))
        for wrong_img, pred_cls, conf in wrong_examples[:number_examples]:
            if not os.path.exists(wrong_pred_folder):
                os.makedirs(wrong_pred_folder)
            base_name = os.path.basename(wrong_img)
            save_path = os.path.join(wrong_pred_folder, f"onnx_pred{pred_cls}_conf{int(conf*100):02d}_{base_name}")
            cv2.imwrite(save_path, cv2.imread(wrong_img))

        # Calculate statistics
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        std_inference_time = np.std(inference_times) * 1000

        print(f"\n" + "="*60)
        print(f"ONNX MODEL RESULTS SUMMARY:")
        print(f"- Model: {os.path.basename(onnx_file_path)}")
        print(f"- Total images: {total_images}")
        print(f"- Correct predictions: {right_counter}")
        print(f"- Accuracy: {right_counter/total_images*100:.2f}%")
        print(f"- Near misses: {near_misses} ({near_misses/total_images*100:.2f}%)")
        print(f"- Combined (correct + near miss): {(right_counter + near_misses)/total_images*100:.2f}%")
        print(f"- Average inference time: {avg_inference_time:.2f} ± {std_inference_time:.2f} ms")
        print(f"- Execution provider: {session.get_providers()[0]}")
        print(f"="*60)

        # Generate and display detailed statistics
        if results_data:
            stats = analyze_prediction_statistics(results_data)
            print_statistics_table(stats)

        return True

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Test ONNX model directly using ONNX Runtime')
    parser.add_argument('onnx_file', help='Path to the ONNX model file')
    parser.add_argument('-t', '--test', nargs='?', const='/home/ubuntu/fotorrojo_ia/test_images/basic/',
                       help='Test folder with JPG images (default: /home/ubuntu/fotorrojo_ia/test_images/basic/)')

    args = parser.parse_args()

    if not args.test:
        print("Error: Test folder is required. Use -t option to specify test images folder.")
        return

    # Test the ONNX model
    success = test_onnx_model(args.onnx_file, args.test)

    if success:
        print("ONNX model testing completed successfully!")
    else:
        print("ONNX model testing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
