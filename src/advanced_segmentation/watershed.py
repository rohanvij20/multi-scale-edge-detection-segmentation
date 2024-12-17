import numpy as np
import cv2
import heapq
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Union


class WatershedSegmentation:
    """Implementation of watershed segmentation algorithm."""

    def __init__(self, output_path: str = "results", min_region_size: int = 100):
        """
        Initialize watershed segmentation.

        Args:
            output_path: Directory for saving results
            min_region_size: Minimum region size to keep
        """
        self.output_path = Path(output_path)
        self.min_region_size = min_region_size
        self._create_output_dirs()

    def _create_output_dirs(self) -> None:
        """Create output directories."""
        dirs = [
            self.output_path / "advanced_segments" / "watershed" / "segments",
            self.output_path / "advanced_segments" / "watershed" / "visualizations",
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _compute_gradient_magnitude(self, gray: np.ndarray) -> np.ndarray:
        """
        Compute gradient magnitude using Sobel operators.

        Args:
            gray: Grayscale input image

        Returns:
            Gradient magnitude image
        """
        h, w = gray.shape

        # Sobel kernels
        sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

        # Initialize gradient images
        gx = np.zeros_like(gray, dtype=np.float32)
        gy = np.zeros_like(gray, dtype=np.float32)

        # Compute gradients
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                region = gray[y - 1 : y + 2, x - 1 : x + 2]
                gx[y, x] = np.sum(region * sx)
                gy[y, x] = np.sum(region * sy)

        return np.sqrt(gx**2 + gy**2)

    def _generate_markers(
        self, gray: np.ndarray, edge_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate markers for watershed segmentation.

        Args:
            gray: Grayscale input image
            edge_map: Optional edge detection result to guide marker placement

        Returns:
            Marker image
        """
        h, w = gray.shape
        markers = np.zeros((h, w), dtype=np.int32)

        # If edge map is provided, use it to avoid placing markers near edges
        if edge_map is not None:
            edge_threshold = np.mean(edge_map) * 0.5
            valid_positions = edge_map < edge_threshold
        else:
            valid_positions = np.ones_like(gray, dtype=bool)

        # Place markers in grid pattern where valid
        step = 30  # Grid spacing
        label = 1
        for y in range(step // 2, h, step):
            for x in range(step // 2, w, step):
                if valid_positions[y, x]:
                    markers[y, x] = label
                    label += 1

        return markers

    def segment(
        self,
        image: np.ndarray,
        edge_map: Optional[np.ndarray] = None,
        image_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Perform watershed segmentation.

        Args:
            image: Input image
            edge_map: Optional edge detection result to guide segmentation
            image_id: Optional image ID for saving results

        Returns:
            Segmented image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Compute gradient magnitude
        gradient = self._compute_gradient_magnitude(gray)

        # Normalize gradient
        mn, mx = gradient.min(), gradient.max()
        if mx - mn < 1e-5:
            gradient[:] = 0
        else:
            gradient = (gradient - mn) / (mx - mn)

        # Generate initial markers
        markers = self._generate_markers(gray, edge_map)
        h, w = gray.shape
        labels = markers.astype(np.int32).copy()

        # Priority queue for watershed
        pq = []
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # Initialize queue with marker positions
        for y in range(h):
            for x in range(w):
                if labels[y, x] > 0:
                    heapq.heappush(pq, (gradient[y, x], y, x))

        # Watershed flooding
        while pq:
            val, cy, cx = heapq.heappop(pq)

            # Check neighbors
            for dy, dx in dirs:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if labels[ny, nx] == 0:
                        labels[ny, nx] = labels[cy, cx]
                        heapq.heappush(pq, (gradient[ny, nx], ny, nx))
                    elif labels[ny, nx] != labels[cy, cx] and labels[ny, nx] != -1:
                        labels[ny, nx] = -1  # Watershed line

        # Create visualization
        segmented = np.zeros_like(image, dtype=np.uint8)
        unique_labels = np.unique(labels)

        # Assign random colors to regions
        np.random.seed(42)  # For reproducibility
        for label in unique_labels:
            if label == -1:  # Watershed line
                segmented[labels == label] = [0, 0, 255]  # Red
            elif label == 0:  # Unlabeled
                segmented[labels == label] = [0, 0, 0]  # Black
            else:
                # Random color for region
                color = np.random.randint(0, 256, size=3, dtype=np.uint8)
                segmented[labels == label] = color

        # Filter small regions
        for label in unique_labels:
            if label > 0:  # Skip watershed lines and background
                region_size = np.sum(labels == label)
                if region_size < self.min_region_size:
                    labels[labels == label] = 0
                    segmented[labels == label] = [0, 0, 0]

        # Save results if image_id is provided
        if image_id is not None:
            # Save segmentation
            seg_path = self.output_path / "advanced_segments" / "watershed" / "segments"
            cv2.imwrite(str(seg_path / f"{image_id}_watershed.png"), segmented)

            # Create and save visualization
            plt.figure(figsize=(15, 5))

            plt.subplot(131)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis("off")

            if edge_map is not None:
                plt.subplot(132)
                plt.imshow(edge_map, cmap="gray")
                plt.title("Edge Map")
                plt.axis("off")

                plt.subplot(133)
                plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
                plt.title("Watershed Segmentation")
                plt.axis("off")
            else:
                plt.subplot(132)
                plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
                plt.title("Watershed Segmentation")
                plt.axis("off")

            vis_path = (
                self.output_path / "advanced_segments" / "watershed" / "visualizations"
            )
            plt.savefig(str(vis_path / f"{image_id}_visualization.png"))
            plt.close()

        return segmented, labels  # Return both colored result and labels


if __name__ == "__main__":
    # Example usage
    import sys

    # Initialize segmenter
    segmenter = WatershedSegmentation()

    # Load and process test image
    image_path = "data/BSDS300/BSDS300-images/images/test/3096.jpg"
    image = cv2.imread(image_path)

    if image is not None:
        segmented, labels = segmenter.segment(image, image_id="3096")
        print(f"Found {len(np.unique(labels))} regions")
    else:
        print(f"Could not load image: {image_path}")
