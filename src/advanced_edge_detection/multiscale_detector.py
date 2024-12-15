import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Union
import matplotlib.pyplot as plt
import pandas as pd


class MultiscaleDetector:
    """Multi-scale edge detection implementation."""

    def __init__(
        self,
        scales: List[float] = None,
        gaussian_sizes: List[int] = None,
        gaussian_sigmas: List[float] = None,
        output_path: str = "results",
    ):
        """
        Initialize the multi-scale detector.

        Args:
            scales: Scales for edge detection [0.5, 1.0, 2.0]
            gaussian_sizes: Gaussian kernel sizes [3, 5, 7]
            gaussian_sigmas: Gaussian sigmas [1.0, 1.4, 2.0]
            output_path: Directory for results
        """
        self.scales = scales or [0.5, 1.0, 2.0]
        self.gaussian_sizes = gaussian_sizes or [3, 5, 7]
        self.gaussian_sigmas = gaussian_sigmas or [1.0, 1.4, 2.0]
        self.output_path = Path(output_path)

        # Validate parameters
        if not (
            len(self.scales) == len(self.gaussian_sizes) == len(self.gaussian_sigmas)
        ):
            raise ValueError(
                "scales, gaussian_sizes, and gaussian_sigmas must have same length"
            )

        # Create output directories
        self._create_output_dirs()

    def _create_output_dirs(self) -> None:
        """Create output directories for results."""
        dirs = [
            self.output_path / "advanced_edges" / "multiscale" / "edges",
            self.output_path / "advanced_edges" / "multiscale" / "visualizations",
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Scale image by given factor."""
        if scale == 1.0:
            return image.copy()
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    def _detect_at_scale(
        self, image: np.ndarray, gaussian_size: int, gaussian_sigma: float
    ) -> np.ndarray:
        """
        Detect edges at specific scale.

        Args:
            image: Input image
            gaussian_size: Size of Gaussian kernel
            gaussian_sigma: Sigma for Gaussian blur

        Returns:
            Edge magnitudes
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (gaussian_size, gaussian_size), gaussian_sigma)

        # Compute gradients
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        # Compute magnitude and normalize
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

        return magnitude

    def detect(self, image: np.ndarray, method: str = "weighted") -> np.ndarray:
        """
        Perform multi-scale edge detection.
        """
        # Add input validation
        if image is None or image.size == 0:
            raise ValueError("Image cannot be empty")
        if len(image.shape) not in [2, 3]:
            raise ValueError("Image must be 2D or 3D array")

        edge_maps = []
        original_size = image.shape[:2]

        # Process each scale
        for scale, g_size, g_sigma in zip(
            self.scales, self.gaussian_sizes, self.gaussian_sigmas
        ):
            # Scale image
            scaled = self._scale_image(image, scale)

            # Detect edges
            edges = self._detect_at_scale(scaled, g_size, g_sigma)

            # Restore original size
            if edges.shape[:2] != original_size:
                edges = cv2.resize(
                    edges,
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            edge_maps.append(edges)

        # Combine results
        if method == "max":
            result = np.maximum.reduce(edge_maps)
        else:  # weighted
            weights = np.array([1.0, 2.0, 1.0])  # Emphasize middle scale
            weights = weights / weights.sum()
            result = np.average(edge_maps, axis=0, weights=weights)

        return result

    def save_visualization(
        self,
        image: np.ndarray,
        result: np.ndarray,
        image_id: str,
        edge_maps: List[np.ndarray],
    ) -> None:
        """Save visualization of results."""
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(151)
        if len(image.shape) == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # Individual scales
        for i, (edges, scale) in enumerate(zip(edge_maps, self.scales)):
            plt.subplot(152 + i)
            plt.imshow(edges, cmap="gray")
            plt.title(f"Scale {scale}")
            plt.axis("off")

        # Final result
        plt.subplot(155)
        plt.imshow(result, cmap="gray")
        plt.title("Combined")
        plt.axis("off")

        # Save
        vis_path = (
            self.output_path
            / "advanced_edges"
            / "multiscale"
            / "visualizations"
            / f"{image_id}_multiscale.png"
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
                result = self.detect(image)

                # Save result
                output_path = (
                    self.output_path
                    / "advanced_edges"
                    / "multiscale"
                    / "edges"
                    / f"{img_path.stem}_edges.png"
                )
                cv2.imwrite(str(output_path), (result * 255).astype(np.uint8))

                # Calculate metrics
                metrics = {
                    "image_id": img_path.stem,
                    "edge_density": np.mean(result > 0.1),
                    "mean_strength": np.mean(result),
                    "max_strength": np.max(result),
                    "std_strength": np.std(result),
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
                self.output_path / "advanced_edges" / "multiscale" / "results.csv",
                index=False,
            )
            return df
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    detector = MultiscaleDetector()
    results_df = detector.process_dataset(
        "data/BSDS300",
        subset="all",  # Process both train and test sets
        n_samples=None,  # Process all images
    )
    print("\nProcessing complete!")
    print(f"Results saved to {detector.output_path}/advanced_edges/multiscale/")
