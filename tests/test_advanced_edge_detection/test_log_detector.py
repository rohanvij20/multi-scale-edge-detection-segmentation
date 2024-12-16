import unittest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil
from src.advanced_edge_detection import LoGDetector


class TestLoGDetector(unittest.TestCase):
    """Test cases for LoGDetector."""

    def setUp(self):
        """Set up test cases."""
        # Create temporary directory for outputs
        self.test_dir = tempfile.mkdtemp()
        self.detector = LoGDetector(output_path=self.test_dir)

        # Create test image with horizontal and vertical edges
        self.test_image = np.zeros((100, 100), dtype=np.uint8)
        self.test_image[40:60, :] = 255  # Horizontal line
        self.test_image[:, 40:60] = 255  # Vertical line

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test detector initialization."""
        # Test default sigma
        detector = LoGDetector()
        self.assertEqual(detector.sigma, 1.0)

        # Test custom sigma
        detector = LoGDetector(sigma=2.0)
        self.assertEqual(detector.sigma, 2.0)

    def test_zero_crossing(self):
        """Test zero crossing detection."""
        # Create test laplacian with known zero crossings
        laplacian = np.array([[-1, -1, 1], [-1, 0, 1], [1, 1, 1]], dtype=np.float32)

        edges = self.detector._zero_crossing(laplacian)
        self.assertTrue(edges[1, 1] > 0)  # Center should be edge point

    def test_detect(self):
        """Test complete edge detection pipeline."""
        # Test with grayscale image
        edges = self.detector.detect(self.test_image)

        self.assertEqual(edges.shape, self.test_image.shape)
        self.assertEqual(edges.dtype, np.uint8)

        # Check if edges are detected
        edge_points = np.count_nonzero(edges)
        self.assertTrue(edge_points > 0)

        # Test with color image
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        color_image[40:60, :] = [255, 0, 0]  # Red horizontal line

        edges = self.detector.detect(color_image)
        self.assertEqual(edges.shape, (100, 100))
        self.assertTrue(np.any(edges > 0))

    def test_process_dataset(self):
        """Test dataset processing functionality."""
        # Create mock dataset structure
        dataset_dir = Path(self.test_dir) / "mock_dataset"
        train_dir = dataset_dir / "BSDS300-images" / "images" / "train"
        test_dir = dataset_dir / "BSDS300-images" / "images" / "test"

        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        # Create sample images
        for i in range(3):
            cv2.imwrite(str(train_dir / f"train_{i}.jpg"), self.test_image)
            cv2.imwrite(str(test_dir / f"test_{i}.jpg"), self.test_image)

        # Test processing
        detector = LoGDetector(output_path=self.test_dir)
        results = detector.process_dataset(dataset_dir, subset="all", n_samples=None)

        self.assertEqual(len(results), 6)  # 3 train + 3 test images

        # Verify required metrics exist
        required_metrics = [
            "image_id",
            "edge_density",
            "mean_strength",
            "max_strength",
            "std_strength",
        ]
        for metric in required_metrics:
            self.assertTrue(metric in results.columns)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty image
        with self.assertRaises(ValueError):
            self.detector.detect(np.array([]))

        # Test with invalid image path
        detector = LoGDetector(output_path=self.test_dir)
        results = detector.process_dataset(self.test_dir / "nonexistent", subset="all")
        self.assertTrue(results.empty)

    def test_visualization(self):
        """Test visualization creation."""
        # Process a test image
        edges = self.detector.detect(self.test_image)
        self.detector.save_visualization(self.test_image, edges, "test_image")

        # Check if visualization was created
        vis_path = (
            Path(self.test_dir)
            / "advanced_edges"
            / "log"
            / "visualizations"
            / "test_image_log.png"
        )
        self.assertTrue(vis_path.exists())


if __name__ == "__main__":
    unittest.main()
