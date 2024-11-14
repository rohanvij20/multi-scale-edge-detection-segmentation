import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import os
import glob


class EdgeBasedSegmentation:
    """Basic implementation of edge-based segmentation using Sobel and Canny detectors."""

    def __init__(self, base_path: str):
        """
        Initialize the segmentation class.

        Args:
            base_path: Base path to the BSDS300 dataset
        """
        self.base_path = Path(base_path)
        self.image = None
        self.edges = None
        self.segments = None
        self.image_id = None

        # Validate dataset structure
        self.validate_dataset_structure()

    def validate_dataset_structure(self):
        """Validate that the BSDS300 dataset structure exists."""
        required_paths = [
            self.base_path / "BSDS300-images" / "images" / "train",
            self.base_path / "BSDS300-images" / "images" / "test",
            self.base_path / "BSDS300-human" / "human" / "color",
            self.base_path / "BSDS300-human" / "human" / "gray",
        ]

        for path in required_paths:
            if not path.exists():
                raise ValueError(f"Required dataset path not found: {path}")

    def get_image_path(self, image_id: str) -> Path:
        """Get the path to an image given its ID."""
        # Check train directory
        train_path = (
            self.base_path / "BSDS300-images" / "images" / "train" / f"{image_id}.jpg"
        )
        if train_path.exists():
            return train_path

        # Check test directory
        test_path = (
            self.base_path / "BSDS300-images" / "images" / "test" / f"{image_id}.jpg"
        )
        if test_path.exists():
            return test_path

        raise ValueError(f"Image {image_id} not found in dataset")

    def get_segmentation_paths(self, image_id: str) -> Dict[str, List[Path]]:
        """Get paths to all ground truth segmentations for an image."""
        seg_paths = {"color": [], "gray": []}

        # Search in all annotator directories for both color and gray
        for mode in ["color", "gray"]:
            base_seg_path = self.base_path / "BSDS300-human" / "human" / mode
            for annotator_dir in base_seg_path.glob(
                "*"
            ):  # iterate through numbered directories
                if annotator_dir.is_dir():
                    seg_file = annotator_dir / f"{image_id}.seg"
                    if seg_file.exists():
                        seg_paths[mode].append(seg_file)

        return seg_paths

    def load_image(self, image_id: str) -> None:
        """Load an image and store its ID."""
        image_path = self.get_image_path(image_id)
        self.image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        self.image_id = image_id

        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")

    def detect_edges_sobel(self, ksize: int = 3) -> np.ndarray:
        """
        Detect edges using Sobel operator.

        Args:
            ksize: Kernel size for Sobel operator (3, 5, 7)
        Returns:
            Edge magnitude image
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # Calculate gradients
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=ksize)

        # Calculate magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize to 0-255 range
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

        self.edges = magnitude
        return magnitude

    def detect_edges_canny(
        self, low_threshold: int = 100, high_threshold: int = 200
    ) -> np.ndarray:
        """
        Detect edges using Canny edge detector.

        Args:
            low_threshold: Lower threshold for hysteresis
            high_threshold: Higher threshold for hysteresis
        Returns:
            Binary edge image
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        self.edges = edges
        return edges

    def segment_image(
        self, threshold: int = 128, min_region_size: int = 100
    ) -> np.ndarray:
        """
        Perform basic segmentation using edge information.

        Args:
            threshold: Threshold for edge strength
            min_region_size: Minimum size of regions to keep
        Returns:
            Labeled image where each segment has a unique integer label
        """
        if self.edges is None:
            raise ValueError("No edges detected. Run edge detection first.")

        # Threshold edges to create binary edge map
        binary_edges = self.edges > threshold

        # Create markers for watershed
        ret, markers = cv2.connectedComponents(~binary_edges.astype(np.uint8))

        # Filter small regions
        unique, counts = np.unique(markers, return_counts=True)
        for label, count in zip(unique, counts):
            if count < min_region_size:
                markers[markers == label] = 0

        self.segments = markers
        return markers

    def save_results(self, output_base_path: str, method: str) -> None:
        """
        Save detection and segmentation results.

        Args:
            output_base_path: Base path for saving results
            method: Name of the method used (e.g., 'sobel', 'canny')
        """
        if self.edges is None or self.segments is None:
            raise ValueError("Run edge detection and segmentation first")

        # Create output directories
        output_base = Path(output_base_path)
        edges_dir = output_base / "edges" / method
        segments_dir = output_base / "segments" / method
        edges_dir.mkdir(parents=True, exist_ok=True)
        segments_dir.mkdir(parents=True, exist_ok=True)

        # Save edge detection result
        edge_vis = cv2.cvtColor(self.edges, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(edges_dir / f"{self.image_id}_edges.jpg"), edge_vis)

        # Save segmentation result
        segment_vis = np.zeros_like(self.image)
        unique_labels = np.unique(self.segments)
        colors = np.random.randint(0, 255, (len(unique_labels), 3))

        for label, color in zip(unique_labels, colors):
            segment_vis[self.segments == label] = color

        cv2.imwrite(str(segments_dir / f"{self.image_id}_segments.jpg"), segment_vis)


def process_dataset_sample(dataset_path: str, output_path: str, n_samples: int = 5):
    """
    Process a sample of images from the dataset.

    Args:
        dataset_path: Path to BSDS300 dataset
        output_path: Path for saving results
        n_samples: Number of sample images to process
    """
    segmenter = EdgeBasedSegmentation(dataset_path)

    # Get list of image IDs from train directory
    train_dir = Path(dataset_path) / "BSDS300-images" / "images" / "train"
    image_ids = [p.stem for p in train_dir.glob("*.jpg")][:n_samples]

    for image_id in image_ids:
        print(f"Processing image {image_id}")

        # Load image
        segmenter.load_image(image_id)

        # Process with Sobel
        segmenter.detect_edges_sobel(ksize=3)
        segmenter.segment_image(threshold=128, min_region_size=100)
        segmenter.save_results(output_path, "sobel")

        # Process with Canny
        segmenter.detect_edges_canny(100, 200)
        segmenter.segment_image(threshold=128, min_region_size=100)
        segmenter.save_results(output_path, "canny")


if __name__ == "__main__":
    # Update these paths to match your setup
    DATASET_PATH = "data/BSDS300"
    OUTPUT_PATH = "results"

    process_dataset_sample(DATASET_PATH, OUTPUT_PATH, n_samples=5)
