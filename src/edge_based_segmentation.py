import collections
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Union
import matplotlib.pyplot as plt
import pandas as pd


class EdgeBasedSegmentation:
    """Enhanced implementation of edge-based segmentation with comparison features."""

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
        self.comparison_metrics = {}

        # Validate dataset structure
        self.validate_dataset_structure()

    def validate_dataset_structure(self) -> None:
        """
        Validate that the BSDS300 dataset structure exists.

        Raises:
            ValueError: If required dataset paths are not found
        """
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
        """
        Get the path to an image given its ID.

        Args:
            image_id: ID of the image to locate

        Returns:
            Path to the image file

        Raises:
            ValueError: If image is not found in dataset
        """
        train_path = (
            self.base_path / "BSDS300-images" / "images" / "train" / f"{image_id}.jpg"
        )
        if train_path.exists():
            return train_path

        test_path = (
            self.base_path / "BSDS300-images" / "images" / "test" / f"{image_id}.jpg"
        )
        if test_path.exists():
            return test_path

        raise ValueError(f"Image {image_id} not found in dataset")

    def load_image(self, image_id: str) -> None:
        """
        Load an image and store its ID.

        Args:
            image_id: ID of the image to load

        Raises:
            ValueError: If image cannot be loaded
        """
        try:
            image_path = self.get_image_path(image_id)
            self.image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            self.image_id = image_id

            if self.image is None:
                raise ValueError(f"Could not load image from {image_path}")
        except Exception as e:
            raise ValueError(f"Error loading image {image_id}: {str(e)}")

    def detect_edges_sobel(self, ksize: int = 3) -> np.ndarray:
        """
        Detect edges using Sobel operator.

        Args:
            ksize: Kernel size for Sobel operator (3, 5, 7)

        Returns:
            Edge magnitude image normalized to [0, 1]

        Raises:
            ValueError: If no image is loaded
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Convert to grayscale
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        edges = np.zeros(image.shape)

        # Define Sobel kernels
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

        # Apply Sobel operator
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                seg = image[y - 1 : y + 2, x - 1 : x + 2]
                gx = np.sum(seg * Gx)
                gy = np.sum(seg * Gy)
                mag = np.sqrt(gx**2 + gy**2)
                edges[y, x] = mag

        # Normalize edges to [0, 1]
        edges = (edges - np.min(edges)) / (np.max(edges) - np.min(edges))

        self.edges = edges
        return edges

    def detect_edges_canny(
        self, low_threshold: int = 10, high_threshold: int = 20
    ) -> np.ndarray:
        """
        Detect edges using Canny algorithm.

        Args:
            low_threshold: Lower threshold for hysteresis
            high_threshold: Higher threshold for hysteresis

        Returns:
            Binary edge image

        Raises:
            ValueError: If no image is loaded
        """
        if self.image is None:
            raise ValueError("No image loaded")

        h, w, _ = self.image.shape
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Gaussian blur
        gauss_kernel = (
            np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
        )
        blurred = np.zeros_like(gray, dtype=np.float32)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                region = gray[y - 1 : y + 2, x - 1 : x + 2]
                blurred[y, x] = np.sum(region * gauss_kernel)

        # Sobel gradients
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

        gradient_magnitude = np.zeros_like(blurred, dtype=np.float32)
        gradient_direction = np.zeros_like(blurred, dtype=np.float32)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                patch = blurred[y - 1 : y + 2, x - 1 : x + 2]
                gx = np.sum(patch * Gx)
                gy = np.sum(patch * Gy)

                mag = np.sqrt(gx**2 + gy**2)
                gradient_magnitude[y, x] = mag

                angle = np.arctan2(gy, gx) * (180.0 / np.pi)  # in degrees
                if angle < 0:
                    angle += 180
                gradient_direction[y, x] = angle

        # Normalize gradient magnitude
        gmin, gmax = gradient_magnitude.min(), gradient_magnitude.max()
        if gmax - gmin > 1e-5:
            gradient_magnitude = (gradient_magnitude - gmin) / (gmax - gmin) * 255
        else:
            gradient_magnitude[:] = 0

        # Non-maximum suppression
        nms = np.zeros_like(gradient_magnitude, dtype=np.float32)
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                angle = gradient_direction[y, x]
                mag = gradient_magnitude[y, x]

                # Find adjacent pixels based on gradient direction
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    q = gradient_magnitude[y, x + 1]
                    r = gradient_magnitude[y, x - 1]
                elif 22.5 <= angle < 67.5:
                    q = gradient_magnitude[y - 1, x + 1]
                    r = gradient_magnitude[y + 1, x - 1]
                elif 67.5 <= angle < 112.5:
                    q = gradient_magnitude[y - 1, x]
                    r = gradient_magnitude[y + 1, x]
                else:
                    q = gradient_magnitude[y - 1, x - 1]
                    r = gradient_magnitude[y + 1, x + 1]

                if mag >= q and mag >= r:
                    nms[y, x] = mag
                else:
                    nms[y, x] = 0

        # Double threshold and hysteresis
        strong = (nms >= high_threshold).astype(np.uint8)
        weak = ((nms >= low_threshold) & (nms < high_threshold)).astype(np.uint8)

        edges = np.zeros_like(strong, dtype=np.uint8)
        visited = np.zeros_like(strong, dtype=np.uint8)

        # Track strong edges
        queue = collections.deque()
        strong_coords = np.argwhere(strong == 1)
        for r, c in strong_coords:
            queue.append((r, c))
            visited[r, c] = 1
            edges[r, c] = 1

        # Connect weak edges to strong edges
        while queue:
            r, c = queue.popleft()
            for nr in range(r - 1, r + 2):
                for nc in range(c - 1, c + 2):
                    if 0 <= nr < h and 0 <= nc < w:
                        if weak[nr, nc] == 1 and visited[nr, nc] == 0:
                            visited[nr, nc] = 1
                            edges[nr, nc] = 1
                            queue.append((nr, nc))

        edges = edges * 255
        self.edges = edges.astype(np.uint8)
        return self.edges

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

        Raises:
            ValueError: If no edges have been detected
        """
        if self.edges is None:
            raise ValueError("No edges detected. Run edge detection first.")

        binary_edges = self.edges > threshold
        ret, markers = cv2.connectedComponents(~binary_edges.astype(np.uint8))

        # Filter small regions
        unique, counts = np.unique(markers, return_counts=True)
        for label, count in zip(unique, counts):
            if count < min_region_size:
                markers[markers == label] = 0

        self.segments = markers
        return markers

    def create_segment_visualization(self, segments: np.ndarray) -> np.ndarray:
        """
        Create a color visualization of segments.

        Args:
            segments: Labeled segmentation image

        Returns:
            Color visualization of segments
        """
        vis = np.zeros_like(self.image)
        unique_labels = np.unique(segments)
        colors = np.random.randint(0, 255, (len(unique_labels), 3))

        for label, color in zip(unique_labels, colors):
            vis[segments == label] = color

        return vis

    def _calculate_metrics(
        self,
        sobel_edges: np.ndarray,
        canny_edges: np.ndarray,
        sobel_segments: np.ndarray,
        canny_segments: np.ndarray,
    ) -> Dict:
        """
        Calculate comparison metrics between Sobel and Canny results.

        Args:
            sobel_edges: Sobel edge detection result
            canny_edges: Canny edge detection result
            sobel_segments: Sobel segmentation result
            canny_segments: Canny segmentation result

        Returns:
            Dictionary containing comparison metrics
        """
        return {
            "sobel_edge_density": np.mean(sobel_edges > 0),
            "canny_edge_density": np.mean(canny_edges > 0),
            "sobel_segments": len(np.unique(sobel_segments)),
            "canny_segments": len(np.unique(canny_segments)),
        }

    def _save_results(
        self,
        output_path: Union[str, Path],
        sobel_edges: np.ndarray,
        canny_edges: np.ndarray,
        sobel_vis: np.ndarray,
        canny_vis: np.ndarray,
    ) -> None:
        """
        Save edge detection and segmentation results.

        Args:
            output_path: Directory to save results
            sobel_edges: Sobel edge detection result
            canny_edges: Canny edge detection result
            sobel_vis: Sobel segmentation visualization
            canny_vis: Canny segmentation visualization
        """
        output_base = Path(output_path)

        # Save Sobel results
        edges_dir = output_base / "edges" / "sobel"
        segments_dir = output_base / "segments" / "sobel"
        edges_dir.mkdir(parents=True, exist_ok=True)
        segments_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(edges_dir / f"{self.image_id}_edges.jpg"), sobel_edges)
        cv2.imwrite(str(segments_dir / f"{self.image_id}_segments.jpg"), sobel_vis)

        # Save Canny results
        edges_dir = output_base / "edges" / "canny"
        segments_dir = output_base / "segments" / "canny"
        edges_dir.mkdir(parents=True, exist_ok=True)
        segments_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(edges_dir / f"{self.image_id}_edges.jpg"), canny_edges)
        cv2.imwrite(str(segments_dir / f"{self.image_id}_segments.jpg"), canny_vis)

    def _create_visualization(
        self,
        metrics: Dict,
        sobel_edges: np.ndarray,
        canny_edges: np.ndarray,
        sobel_vis: np.ndarray,
        canny_vis: np.ndarray,
        show_plot: bool,
        save_plot: bool,
        output_path: Union[str, Path],
    ) -> None:
        """
        Create comparison visualization.

        Args:
            metrics: Dictionary of comparison metrics
            sobel_edges: Sobel edge detection result
            canny_edges: Canny edge detection result
            sobel_vis: Sobel segmentation visualization
            canny_vis: Canny segmentation visualization
            show_plot: Whether to display the plot
            save_plot: Whether to save the plot
            output_path: Directory to save the plot
        """
        plt.figure(figsize=(15, 10))

        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        # Sobel results
        plt.subplot(2, 3, 2)
        plt.imshow(sobel_edges, cmap="gray")
        plt.title(f"Sobel Edges\nDensity: {metrics['sobel_edge_density']:.3f}")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(sobel_vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Sobel Segments\nCount: {metrics['sobel_segments']}")
        plt.axis("off")

        # Canny results
        plt.subplot(2, 3, 5)
        plt.imshow(canny_edges, cmap="gray")
        plt.title(f"Canny Edges\nDensity: {metrics['canny_edge_density']:.3f}")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(canny_vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Canny Segments\nCount: {metrics['canny_segments']}")
        plt.axis("off")

        plt.tight_layout()

        if save_plot:
            comparison_dir = Path(output_path) / "comparisons"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(comparison_dir / f"comparison_{self.image_id}.png"))

        if show_plot:
            plt.show()
        else:
            plt.close()

    def compare_edges(
        self,
        show_plot: bool = True,
        save_plot: bool = True,
        output_path: str = "results",
    ) -> Dict:
        """
        Compare Sobel, Canny, and Multi-scale edge detection results.

        Args:
            show_plot: Whether to display the plot
            save_plot: Whether to save the plot
            output_path: Directory to save results

        Returns:
            Dictionary containing comparison metrics
        """
        if self.image is None:
            raise ValueError("No image loaded")

        output_base = Path(output_path)

        # Process with Sobel
        sobel_edges = self.detect_edges_sobel(ksize=3)
        sobel_segments = self.segment_image(threshold=128, min_region_size=100)
        sobel_vis = self.create_segment_visualization(sobel_segments)

        # Process with Canny
        canny_edges = self.detect_edges_canny(10, 20)
        canny_segments = self.segment_image(threshold=128, min_region_size=100)
        canny_vis = self.create_segment_visualization(canny_segments)

        # Process with Multi-scale detector
        from src.advanced_edge_detection import MultiscaleDetector

        ms_detector = MultiscaleDetector(output_path=output_path)
        ms_edges = ms_detector.detect(self.image)
        self.edges = (ms_edges * 255).astype(np.uint8)  # Set for segmentation
        ms_segments = self.segment_image(threshold=128, min_region_size=100)
        ms_vis = self.create_segment_visualization(ms_segments)

        # Save Sobel results
        edges_dir = output_base / "edges" / "sobel"
        segments_dir = output_base / "segments" / "sobel"
        edges_dir.mkdir(parents=True, exist_ok=True)
        segments_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(edges_dir / f"{self.image_id}_edges.jpg"), sobel_edges)
        cv2.imwrite(str(segments_dir / f"{self.image_id}_segments.jpg"), sobel_vis)

        # Save Canny results
        edges_dir = output_base / "edges" / "canny"
        segments_dir = output_base / "segments" / "canny"
        edges_dir.mkdir(parents=True, exist_ok=True)
        segments_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(edges_dir / f"{self.image_id}_edges.jpg"), canny_edges)
        cv2.imwrite(str(segments_dir / f"{self.image_id}_segments.jpg"), canny_vis)

        # Calculate comparison metrics
        metrics = {
            "sobel_edge_density": np.mean(sobel_edges > 0),
            "canny_edge_density": np.mean(canny_edges > 0),
            "multiscale_edge_density": np.mean(ms_edges > 0.1),
            "sobel_segments": len(np.unique(sobel_segments)),
            "canny_segments": len(np.unique(canny_segments)),
            "multiscale_segments": len(np.unique(ms_segments)),
        }

        if show_plot or save_plot:
            plt.figure(figsize=(20, 10))

            # Original image
            plt.subplot(2, 3, 1)
            plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis("off")

            # Sobel results
            plt.subplot(2, 3, 2)
            plt.imshow(sobel_edges, cmap="gray")
            plt.title(f"Sobel Edges\nDensity: {metrics['sobel_edge_density']:.3f}")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(cv2.cvtColor(sobel_vis, cv2.COLOR_BGR2RGB))
            plt.title(f"Sobel Segments\nCount: {metrics['sobel_segments']}")
            plt.axis("off")

            # Canny results
            plt.subplot(2, 3, 4)
            plt.imshow(canny_edges, cmap="gray")
            plt.title(f"Canny Edges\nDensity: {metrics['canny_edge_density']:.3f}")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(cv2.cvtColor(canny_vis, cv2.COLOR_BGR2RGB))
            plt.title(f"Canny Segments\nCount: {metrics['canny_segments']}")
            plt.axis("off")

            # Multi-scale results
            plt.subplot(2, 3, 6)
            plt.imshow(ms_edges, cmap="gray")
            plt.title(
                f"Multi-scale Edges\nDensity: {metrics['multiscale_edge_density']:.3f}"
            )
            plt.axis("off")

            plt.tight_layout()

            if save_plot:
                comparison_dir = Path("results") / "comparisons"
                comparison_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(comparison_dir / f"comparison_{self.image_id}.png"))

            if show_plot:
                plt.show()
            else:
                plt.close()

        return metrics


def process_dataset_sample(
    dataset_path: str,
    output_path: str,
    n_samples: Optional[int] = None,
    include_test: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a sample of images from the dataset with comprehensive comparison.

    Args:
        dataset_path: Path to BSDS300 dataset
        output_path: Path for saving results
        n_samples: Number of sample images to process (None for all)
        include_test: Whether to include test set images

    Returns:
        Tuple containing (results DataFrame, summary DataFrame)
    """
    segmenter = EdgeBasedSegmentation(dataset_path)
    results = []

    # Create all necessary directories
    output_base = Path(output_path)
    directories = [
        output_base / "edges" / "sobel",
        output_base / "edges" / "canny",
        output_base / "segments" / "sobel",
        output_base / "segments" / "canny",
        output_base / "comparisons",
    ]
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Get training images
    train_dir = Path(dataset_path) / "BSDS300-images" / "images" / "train"
    train_ids = [p.stem for p in train_dir.glob("*.jpg")]
    print(f"Found {len(train_ids)} training images")

    # Get test images if requested
    test_ids = []
    if include_test:
        test_dir = Path(dataset_path) / "BSDS300-images" / "images" / "test"
        test_ids = [p.stem for p in test_dir.glob("*.jpg")]
        print(f"Found {len(test_ids)} test images")

    # Combine and limit samples if specified
    image_ids = train_ids + test_ids
    if n_samples is not None:
        image_ids = image_ids[:n_samples]
        print(f"Processing {n_samples} samples")
    else:
        print(f"Processing all {len(image_ids)} images")

    # Process each image
    total_images = len(image_ids)
    for idx, image_id in enumerate(image_ids, 1):
        print(f"Processing image {image_id} ({idx}/{total_images})")
        try:
            segmenter.load_image(image_id)

            # Run comparison and collect metrics
            metrics = segmenter.compare_edges(
                show_plot=False, save_plot=True, output_path=output_path
            )
            metrics["image_id"] = image_id
            metrics["dataset"] = "test" if image_id in test_ids else "train"
            metrics["image_height"], metrics["image_width"] = segmenter.image.shape[:2]

            results.append(metrics)

        except Exception as e:
            print(f"Error processing image {image_id}: {str(e)}")
            continue

    # Create summary report
    if results:
        df = pd.DataFrame(results)

        # Calculate additional statistics
        df["edge_density_difference"] = (
            df["canny_edge_density"] - df["sobel_edge_density"]
        )
        df["segment_count_difference"] = df["canny_segments"] - df["sobel_segments"]

        summary = df.describe()

        # Save results
        results_dir = Path(output_path)
        results_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(results_dir / "edge_detection_comparison.csv", index=False)

        with open(results_dir / "summary_report.txt", "w") as f:
            f.write("Edge Detection Comparison Summary\n")
            f.write("================================\n\n")

            f.write("Dataset Statistics:\n")
            f.write(f"Total images processed: {len(results)}\n")
            f.write(
                f"Training images: {len([r for r in results if r['dataset'] == 'train'])}\n"
            )
            f.write(
                f"Test images: {len([r for r in results if r['dataset'] == 'test'])}\n\n"
            )

            f.write("Metrics Summary:\n")
            f.write(str(summary))
            f.write("\n\n")

            f.write("Method Comparison:\n")
            f.write(
                f"Average Sobel edge density: {df['sobel_edge_density'].mean():.3f}\n"
            )
            f.write(
                f"Average Canny edge density: {df['canny_edge_density'].mean():.3f}\n"
            )
            f.write(f"Average Sobel segments: {df['sobel_segments'].mean():.1f}\n")
            f.write(f"Average Canny segments: {df['canny_segments'].mean():.1f}\n")

            f.write("\nKey Observations:\n")
            f.write(
                f"- Edge density difference (Canny - Sobel): {df['edge_density_difference'].mean():.3f}\n"
            )
            f.write(
                f"- Segment count difference (Canny - Sobel): {df['segment_count_difference'].mean():.1f}\n"
            )

        print(f"\nResults saved to {results_dir}")
        return df, summary
    else:
        print("No results generated.")
        return None, None


if __name__ == "__main__":
    DATASET_PATH = "data/BSDS300"
    OUTPUT_PATH = "results"

    # Process all images from both training and test sets
    df, summary = process_dataset_sample(
        DATASET_PATH,
        OUTPUT_PATH,
        n_samples=None,  # Process all images
        include_test=True,  # Include test set
    )

    if df is not None:
        print("\nSummary Statistics:")
        print(summary)
