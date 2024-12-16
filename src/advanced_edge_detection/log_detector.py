import numpy as np
import cv2
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import matplotlib.pyplot as plt


class LoGDetector:
    """Laplacian of Gaussian edge detector implementation."""

    def __init__(self, sigma: float = 1.0, output_path: str = "results"):
        """
        Initialize LoG detector.

        Args:
            sigma: Standard deviation for Gaussian kernel
            output_path: Directory for saving results
        """
        self.sigma = sigma
        self.output_path = Path(output_path)
        self._create_output_dirs()

    def _create_output_dirs(self) -> None:
        """Create output directories for results."""
        dirs = [
            self.output_path / "advanced_edges" / "log" / "edges",
            self.output_path / "advanced_edges" / "log" / "visualizations",
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply LoG edge detection.

        Args:
            image: Input image

        Returns:
            Binary edge image
        """
        if image is None or image.size == 0:
            raise ValueError("Image cannot be empty")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Create Gaussian kernel
        kernel_size = int(6 * self.sigma + 1)
        gaussian_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        mid = kernel_size // 2
        sigma_squared_2 = 2 * (self.sigma**2)

        # Fill Gaussian kernel
        for r in range(kernel_size):
            for c in range(kernel_size):
                y = r - mid
                x = c - mid
                gaussian_kernel[r, c] = np.exp(-(x**2 + y**2) / sigma_squared_2)

        # Normalize kernel
        gaussian_kernel /= gaussian_kernel.sum()

        # Apply Gaussian blur
        blurred = self._convolve(gray, gaussian_kernel)

        # Apply Laplacian
        laplacian_kernel = np.array(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32
        )
        laplacian = self._convolve(blurred, laplacian_kernel)

        # Detect zero crossings
        edges = self._zero_crossing(laplacian)

        return edges

    def _convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Apply convolution with given kernel.

        Args:
            image: Input image
            kernel: Convolution kernel

        Returns:
            Convolved image
        """
        h, w = image.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Create padded image
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        output = np.zeros_like(image, dtype=np.float32)

        # Apply convolution
        for y in range(h):
            for x in range(w):
                region = padded[y : y + kh, x : x + kw]
                output[y, x] = np.sum(region * kernel)

        return output

    def _zero_crossing(self, laplacian: np.ndarray) -> np.ndarray:
        """
        Detect zero crossings in LoG response.

        Args:
            laplacian: LoG filtered image

        Returns:
            Binary edge image
        """
        h, w = laplacian.shape
        edges = np.zeros((h, w), dtype=np.uint8)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                # Check for sign changes in 3x3 neighborhood
                region = laplacian[y - 1 : y + 2, x - 1 : x + 2]
                if (region.min() < 0) and (region.max() > 0):
                    edges[y, x] = 255

        return edges

    def save_visualization(
        self, image: np.ndarray, edges: np.ndarray, image_id: str
    ) -> None:
        """Save visualization of results."""
        plt.figure(figsize=(10, 5))

        # Original image
        plt.subplot(121)
        if len(image.shape) == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # Edge detection result
        plt.subplot(122)
        plt.imshow(edges, cmap="gray")
        plt.title(f"LoG Edges (σ={self.sigma})")
        plt.axis("off")

        # Save
        vis_path = (
            self.output_path
            / "advanced_edges"
            / "log"
            / "visualizations"
            / f"{image_id}_log.png"
        )
        plt.savefig(vis_path)
        plt.close()

    def process_dataset(
        self,
        dataset_path: Union[str, Path],
        subset: str = "all",
        n_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Process images from dataset.

        Args:
            dataset_path: Path to BSDS300 dataset
            subset: 'train', 'test', or 'all'
            n_samples: Number of images to process (None for all)

        Returns:
            DataFrame with results
        """
        dataset_path = Path(dataset_path)
        results = []

        # Get image paths
        paths = []
        if subset in ["train", "all"]:
            paths.extend(
                list(
                    (dataset_path / "BSDS300-images" / "images" / "train").glob("*.jpg")
                )
            )
        if subset in ["test", "all"]:
            paths.extend(
                list(
                    (dataset_path / "BSDS300-images" / "images" / "test").glob("*.jpg")
                )
            )

        if n_samples:
            paths = paths[:n_samples]

        print(f"Processing {len(paths)} images...")

        # Process each image
        for idx, img_path in enumerate(paths, 1):
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"Error loading {img_path}")
                    continue

                # Detect edges
                edges = self.detect(image)

                # Save result
                output_path = (
                    self.output_path
                    / "advanced_edges"
                    / "log"
                    / "edges"
                    / f"{img_path.stem}_edges.png"
                )
                cv2.imwrite(str(output_path), edges)

                # Save visualization
                self.save_visualization(image, edges, img_path.stem)

                # Calculate metrics
                metrics = {
                    "image_id": img_path.stem,
                    "edge_density": np.mean(edges > 0),
                    "mean_strength": np.mean(edges),
                    "max_strength": np.max(edges),
                    "std_strength": np.std(edges),
                }
                results.append(metrics)

                print(f"Processed {idx}/{len(paths)}: {img_path.stem}")

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

        # Create summary DataFrame
        if results:
            df = pd.DataFrame(results)
            df.to_csv(
                self.output_path / "advanced_edges" / "log" / "results.csv", index=False
            )
            return df
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    detector = LoGDetector(sigma=1.0)
    results_df = detector.process_dataset(
        "data/BSDS300",
        subset="all",  # Process both train and test sets
        n_samples=None,  # Process all images
    )
    print("\nProcessing complete!")
    print(f"Results saved to {detector.output_path}/advanced_edges/log/")
