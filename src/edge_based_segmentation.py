import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class EdgeBasedSegmentation:
    """Basic implementation of edge-based segmentation using Sobel and Canny detectors."""

    def __init__(self):
        self.image = None
        self.edges = None
        self.segments = None

    def load_image(self, image_path: str) -> None:
        """Load an image from the specified path."""
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)
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

    def visualize_results(
        self, output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualize edge detection and segmentation results.

        Args:
            output_path: Optional path to save visualization
        Returns:
            Tuple of (edge visualization, segmentation visualization)
        """
        if self.edges is None or self.segments is None:
            raise ValueError("Run edge detection and segmentation first")

        # Edge visualization
        edge_vis = cv2.cvtColor(self.edges, cv2.COLOR_GRAY2BGR)

        # Segmentation visualization
        segment_vis = np.zeros_like(self.image)
        unique_labels = np.unique(self.segments)
        colors = np.random.randint(0, 255, (len(unique_labels), 3))

        for label, color in zip(unique_labels, colors):
            segment_vis[self.segments == label] = color

        if output_path:
            cv2.imwrite(f"{output_path}_edges.jpg", edge_vis)
            cv2.imwrite(f"{output_path}_segments.jpg", segment_vis)

        return edge_vis, segment_vis


def main():
    """Example usage of the EdgeBasedSegmentation class."""
    segmenter = EdgeBasedSegmentation()

    # Load image
    image_path = "path/to/your/image.jpg"  # Update with actual image path
    segmenter.load_image(image_path)

    # Detect edges using both methods
    edges_sobel = segmenter.detect_edges_sobel(ksize=3)
    edges_canny = segmenter.detect_edges_canny(100, 200)

    # Perform segmentation using Canny edges
    segments = segmenter.segment_image(threshold=128, min_region_size=100)

    # Visualize results
    edge_vis, segment_vis = segmenter.visualize_results("output")


if __name__ == "__main__":
    main()
