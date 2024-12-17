import numpy as np
import cv2
from collections import deque
from typing import Tuple, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt


class RegionGrowing:
    """Advanced region growing segmentation implementation."""

    def __init__(
        self,
        threshold: float = 0.05,
        num_seeds: int = 100,
        min_region_size: int = 100,
        output_path: str = "results",
    ):
        """
        Initialize region growing segmentation.

        Args:
            threshold: Intensity similarity threshold
            num_seeds: Number of seed points
            min_region_size: Minimum region size to keep
            output_path: Directory for saving results
        """
        self.threshold = threshold
        self.num_seeds = num_seeds
        self.min_region_size = min_region_size
        self.output_path = Path(output_path)
        self._create_output_dirs()

    def _create_output_dirs(self) -> None:
        """Create output directories."""
        dirs = [
            self.output_path / "advanced_segments" / "region_growing" / "segments",
            self.output_path
            / "advanced_segments"
            / "region_growing"
            / "visualizations",
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _check_point(
        self,
        curr_point: Tuple[int, int],
        next_point: Tuple[int, int],
        queue: deque,
        region: np.ndarray,
        visited: np.ndarray,
        image: np.ndarray,
        color: int,
    ) -> None:
        """Check if a point should be added to the region."""
        x0, y0 = curr_point
        x1, y1 = next_point
        h, w = image.shape

        if (0 <= x1 < w) and (0 <= y1 < h):
            region[y1, x1] = color
            if not visited[y1, x1]:
                visited[y1, x1] = 1
                if np.abs(image[y0, x0] - image[y1, x1]) < self.threshold:
                    queue.append((x1, y1))

    def _grow_region(
        self,
        seed: Tuple[int, int],
        image: np.ndarray,
        region: np.ndarray,
        visited: np.ndarray,
        color: int,
    ) -> None:
        """Grow region from seed point."""
        queue = deque([seed])

        while queue:
            curr = queue.popleft()
            x, y = curr

            # Check 4-connected neighbors
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

            for next_point in neighbors:
                self._check_point(
                    curr, next_point, queue, region, visited, image, color
                )

    def segment(
        self,
        image: np.ndarray,
        edge_map: Optional[np.ndarray] = None,
        image_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Perform region growing segmentation.

        Args:
            image: Input image
            edge_map: Optional edge detection result to guide segmentation
            image_id: Optional image ID for saving results

        Returns:
            Segmented image
        """
        # Convert to grayscale and preprocess exactly like original
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert to float32 and remove border pixels as in original
        image = image.astype(np.float32)[1:-1, 1:-1]

        # Normalize exactly like original
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Get dimensions after border removal
        h, w = image.shape
        print(f"Image range: max={np.max(image)}, min={np.min(image)}")

        # Initialize arrays exactly like original
        region = np.zeros((h, w), dtype=np.uint8)
        visited = np.zeros((h, w))

        # Use random.randint as in original instead of np.random
        import random

        for i in range(self.num_seeds):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            seed = (x, y)
            color = int(i * 255 / self.num_seeds)  # Same color calculation as original

            # Use deque for queue as in original
            Q = deque()
            Q.append(seed)

            while Q:
                curr = Q.popleft()
                x, y = curr

                # Check 4-connected neighbors exactly as in original
                neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

                for next_point in neighbors:
                    x1, y1 = next_point
                    if (x1 >= 0 and x1 < w) and (y1 >= 0 and y1 < h):
                        region[y1, x1] = color
                        if not visited[y1, x1]:
                            visited[y1, x1] = 1
                            if np.abs(image[y, x] - image[y1, x1]) < self.threshold:
                                Q.append((x1, y1))

        # Save results if image_id is provided
        if image_id is not None:
            seg_path = (
                self.output_path / "advanced_segments" / "region_growing" / "segments"
            )
            vis_path = (
                self.output_path
                / "advanced_segments"
                / "region_growing"
                / "visualizations"
            )

            # Save segmentation result
            cv2.imwrite(str(seg_path / f"{image_id}_segments.png"), region)

            # Create visualization as in original
            plt.figure(figsize=(15, 5))

            plt.subplot(131)
            plt.imshow(image * 255, vmin=0, vmax=255)  # Scale like original
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(132)
            plt.imshow(region, vmin=0, vmax=255)  # Same scale as original
            plt.title("Region Growing")
            plt.axis("off")

            if edge_map is not None:
                plt.subplot(133)
                plt.imshow(edge_map, cmap="gray")
                plt.title("Edge Map")
                plt.axis("off")

            plt.savefig(str(vis_path / f"{image_id}_visualization.png"))
            plt.close()

        return region

    def process_dataset(
        self,
        dataset_path: str,
        edge_maps_path: Optional[str] = None,
        n_samples: Optional[int] = None,
    ) -> dict:
        """
        Process entire dataset.

        Args:
            dataset_path: Path to image dataset
            edge_maps_path: Optional path to edge detection results
            n_samples: Number of samples to process (None for all)

        Returns:
            Dictionary with segmentation metrics
        """
        dataset_path = Path(dataset_path)
        results = []

        # Get image paths
        image_paths = list(dataset_path.glob("**/*.jpg"))
        if n_samples:
            image_paths = image_paths[:n_samples]

        print(f"Processing {len(image_paths)} images...")

        for idx, img_path in enumerate(image_paths, 1):
            try:
                # Load image
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

                # Load edge map if available
                edge_map = None
                if edge_maps_path:
                    edge_path = Path(edge_maps_path) / f"{img_path.stem}_edges.png"
                    if edge_path.exists():
                        edge_map = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)

                # Process image
                segments = self.segment(image, edge_map, img_path.stem)

                print(f"Processed {idx}/{len(image_paths)}: {img_path.stem}")

            except Exception as e:
                print(f"Error processing {img_path.stem}: {str(e)}")
                continue

        return results


if __name__ == "__main__":
    # Example usage
    segmenter = RegionGrowing(threshold=0.05, num_seeds=100)
    results = segmenter.process_dataset(
        "data/BSDS300/BSDS300-images/images/train",
        "results/edges/sobel",
        n_samples=None,
    )
