#!/usr/bin/env python3
"""
Image Size Analyzer

This program walks through a folder structure and analyzes the sizes of all images found.
It generates statistics about image dimensions without loading the full image data.

Features:
- Fast image size detection using PIL header reading
- Statistics by image size and by folder
- Support for common image formats (jpg, jpeg, png, bmp, tiff, webp)
- Command-line interface with customizable output
"""

import os
import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path
from PIL import Image
import time


class ImageSizeAnalyzer:
    """Analyzes image sizes in a folder structure efficiently"""

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    def __init__(self, root_folder):
        self.root_folder = Path(root_folder)
        self.image_sizes = []  # List of (width, height, filepath)
        self.folder_stats = defaultdict(list)  # folder -> list of (width, height)
        self.errors = []  # List of files that couldn't be processed

    def get_image_size_fast(self, image_path):
        """Get image size without loading the full image into memory"""
        try:
            with Image.open(image_path) as img:
                return img.size  # Returns (width, height)
        except Exception as e:
            self.errors.append((image_path, str(e)))
            return None

    def walk_and_analyze(self, progress_callback=None):
        """Walk through the folder structure and analyze all images"""
        print(f"Analyzing images in: {self.root_folder}")
        print(f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}")

        total_files = 0
        processed_files = 0
        start_time = time.time()

        # First pass: count total image files for progress tracking
        for root, dirs, files in os.walk(self.root_folder):
            for file in files:
                if Path(file).suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    total_files += 1

        print(f"Found {total_files} image files to analyze...")

        # Second pass: analyze images
        for root, dirs, files in os.walk(self.root_folder):
            root_path = Path(root)
            folder_relative = root_path.relative_to(self.root_folder)

            for file in files:
                file_path = root_path / file

                # Check if it's a supported image format
                if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    processed_files += 1

                    # Get image size efficiently
                    size = self.get_image_size_fast(file_path)
                    if size:
                        width, height = size
                        self.image_sizes.append((width, height, str(file_path)))
                        self.folder_stats[str(folder_relative)].append((width, height))

                    # Progress reporting
                    if progress_callback and processed_files % 100 == 0:
                        progress = (processed_files / total_files) * 100
                        progress_callback(processed_files, total_files, progress)

        elapsed_time = time.time() - start_time
        print(f"\nAnalysis complete!")
        print(f"Processed {processed_files} images in {elapsed_time:.2f} seconds")
        print(f"Average: {processed_files/elapsed_time:.1f} images/second")

        if self.errors:
            print(f"Errors encountered: {len(self.errors)} files")

    def get_size_statistics(self):
        """Generate statistics about image sizes"""
        if not self.image_sizes:
            return {}

        # Count occurrences of each size
        size_counter = Counter((width, height) for width, height, _ in self.image_sizes)

        # Calculate statistics
        widths = [width for width, height, _ in self.image_sizes]
        heights = [height for width, height, _ in self.image_sizes]

        stats = {
            'total_images': len(self.image_sizes),
            'unique_sizes': len(size_counter),
            'size_distribution': dict(size_counter),
            'width_stats': {
                'min': min(widths),
                'max': max(widths),
                'avg': sum(widths) / len(widths)
            },
            'height_stats': {
                'min': min(heights),
                'max': max(heights),
                'avg': sum(heights) / len(heights)
            }
        }

        return stats

    def get_folder_statistics(self):
        """Generate statistics by folder"""
        folder_stats = {}

        for folder, sizes in self.folder_stats.items():
            if not sizes:
                continue

            size_counter = Counter(sizes)
            widths = [width for width, height in sizes]
            heights = [height for width, height in sizes]

            folder_stats[folder] = {
                'image_count': len(sizes),
                'unique_sizes': len(size_counter),
                'size_distribution': {f"{w}x{h}": count for (w, h), count in size_counter.items()},
                'width_range': f"{min(widths)}-{max(widths)}",
                'height_range': f"{min(heights)}-{max(heights)}"
            }

        return folder_stats

    def print_size_report(self, top_n=10):
        """Print a formatted report of image sizes"""
        stats = self.get_size_statistics()

        print("\n" + "="*60)
        print("IMAGE SIZE ANALYSIS REPORT")
        print("="*60)

        print(f"Total images analyzed: {stats['total_images']}")
        print(f"Unique image sizes: {stats['unique_sizes']}")

        print(f"\nWidth statistics:")
        print(f"  Min: {stats['width_stats']['min']}px")
        print(f"  Max: {stats['width_stats']['max']}px")
        print(f"  Avg: {stats['width_stats']['avg']:.1f}px")

        print(f"\nHeight statistics:")
        print(f"  Min: {stats['height_stats']['min']}px")
        print(f"  Max: {stats['height_stats']['max']}px")
        print(f"  Avg: {stats['height_stats']['avg']:.1f}px")

        print(f"\nTop {top_n} most common image sizes:")
        size_items = sorted(stats['size_distribution'].items(), key=lambda x: x[1], reverse=True)
        for i, ((width, height), count) in enumerate(size_items[:top_n], 1):
            percentage = (count / stats['total_images']) * 100
            print(f"  {i:2d}. {width}x{height} - {count} images ({percentage:.1f}%)")

    def print_folder_report(self):
        """Print a formatted report by folder"""
        folder_stats = self.get_folder_statistics()

        print("\n" + "="*60)
        print("FOLDER-WISE ANALYSIS")
        print("="*60)

        for folder, stats in sorted(folder_stats.items()):
            print(f"\nFolder: {folder}")
            print(f"  Images: {stats['image_count']}")
            print(f"  Unique sizes: {stats['unique_sizes']}")
            print(f"  Width range: {stats['width_range']}px")
            print(f"  Height range: {stats['height_range']}px")

            # Show top 3 sizes in this folder
            size_items = sorted(stats['size_distribution'].items(), key=lambda x: x[1], reverse=True)
            if size_items:
                print(f"  Most common sizes:")
                for size, count in size_items[:3]:
                    print(f"    {size}: {count} images")

    def save_detailed_report(self, output_file):
        """Save detailed statistics to a JSON file"""
        report = {
            'analysis_info': {
                'root_folder': str(self.root_folder),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_images': len(self.image_sizes),
                'errors': len(self.errors)
            },
            'size_statistics': self.get_size_statistics(),
            'folder_statistics': self.get_folder_statistics(),
            'errors': self.errors[:100]  # Limit error list to first 100
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed report saved to: {output_file}")


def progress_callback(processed, total, percentage):
    """Simple progress callback"""
    print(f"\rProgress: {processed}/{total} ({percentage:.1f}%)", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze image sizes in a folder structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_size_analyzer.py ./images
  python image_size_analyzer.py C:/photos --top 20 --output report.json
  python image_size_analyzer.py /data/images --no-folder-stats
        """
    )

    parser.add_argument('folder', help='Root folder to analyze')
    parser.add_argument('--top', '-t', type=int, default=10,
                       help='Number of top sizes to show (default: 10)')
    parser.add_argument('--output', '-o', help='Save detailed report to JSON file')
    parser.add_argument('--no-folder-stats', action='store_true',
                       help='Skip folder-wise statistics')

    args = parser.parse_args()

    # Validate input folder
    if not os.path.exists(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist")
        return 1

    if not os.path.isdir(args.folder):
        print(f"Error: '{args.folder}' is not a directory")
        return 1

    # Create analyzer and run analysis
    analyzer = ImageSizeAnalyzer(args.folder)
    analyzer.walk_and_analyze(progress_callback)

    # Generate reports
    analyzer.print_size_report(args.top)

    if not args.no_folder_stats:
        analyzer.print_folder_report()

    if args.output:
        analyzer.save_detailed_report(args.output)

    if analyzer.errors:
        print(f"\nNote: {len(analyzer.errors)} files could not be processed")
        print("Use --output option to see detailed error list")

    return 0


if __name__ == "__main__":
    exit(main())
