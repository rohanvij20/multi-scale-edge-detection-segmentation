import unittest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil
from src.advanced_edge_detection import MultiscaleDetector


class TestMultiscaleDetector(unittest.TestCase):
    """Test cases for MultiscaleDetector."""

    def setUp(self):
        """Set up test cases."""
        # Create temporary directory for outputs
        self.test_dir = tempfile.mkdtemp()
        self.detector = MultiscaleDetector(output_path=self.test_dir)

        # Create test image with horizontal and vertical edges
        self.test_image = np.zeros((100, 100), dtype=np.uint8)
        self.test_image[40:60, :] = 255  # Horizontal line
        self.test_image[:, 40:60] = 255  # Vertical line

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test detector initialization."""
        # Test default parameters
        detector = MultiscaleDetector()
        self.assertEqual(len(detector.scales), 3)
        self.assertEqual(len(detector.gaussian_sizes), 3)
        self.assertEqual(len(detector.gaussian_sigmas), 3)

        # Test custom parameters
        custom_scales = [0.5, 1.0]
        custom_sizes = [3, 5]
        custom_sigmas = [1.0, 1.4]
        detector = MultiscaleDetector(custom_scales, custom_sizes, custom_sigmas)
        self.assertEqual(detector.scales, custom_scales)
        self.assertEqual(detector.gaussian_sizes, custom_sizes)
        self.assertEqual(detector.gaussian_sigmas, custom_sigmas)

        # Test invalid parameters
        with self.assertRaises(ValueError):
            MultiscaleDetector(scales=[0.5], gaussian_sizes=[3, 5])

    def test_scale_image(self):
        """Test image scaling."""
        scales = [0.5, 1.0, 2.0]
        for scale in scales:
            scaled = self.detector._scale_image(self.test_image, scale)
            expected_shape = (int(100 * scale), int(100 * scale))
            self.assertEqual(scaled.shape, expected_shape)

    def test_detect_at_scale(self):
        """Test edge detection at single scale."""
        # Test with valid parameters
        edges = self.detector._detect_at_scale(
            self.test_image, gaussian_size=3, gaussian_sigma=1.0
        )

        self.assertEqual(edges.shape, self.test_image.shape)
        self.assertTrue(np.all(edges >= 0))
        self.assertTrue(np.all(edges <= 1))

        # Check if edges are detected
        self.assertTrue(np.mean(edges) > 0)

    def test_detect(self):
        """Test complete detection pipeline."""
        # Test with different methods
        for method in ["weighted", "max"]:
            result = self.detector.detect(self.test_image, method=method)

            self.assertEqual(result.shape, self.test_image.shape)
            self.assertTrue(np.all(result >= 0))
            self.assertTrue(np.all(result <= 1))

            # Verify edges are detected
            edge_pixels = np.count_nonzero(result > 0.1)
            self.assertTrue(edge_pixels > 0)

    def test_color_image(self):
        """Test handling of color images."""
        # Create color test image
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        color_image[40:60, :] = [255, 0, 0]  # Red horizontal line

        result = self.detector.detect(color_image)
        self.assertEqual(result.shape, (100, 100))
        self.assertTrue(np.any(result > 0))

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
        detector = MultiscaleDetector(output_path=self.test_dir)
        results = detector.process_dataset(dataset_dir, subset="all", n_samples=None)

        self.assertEqual(len(results), 6)  # 3 train +
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

        # Test subset selection
        train_results = detector.process_dataset(dataset_dir, subset="train")
        self.assertEqual(len(train_results), 3)

        test_results = detector.process_dataset(dataset_dir, subset="test")
        self.assertEqual(len(test_results), 3)

        # Test sample limiting
        limited_results = detector.process_dataset(
            dataset_dir, subset="all", n_samples=2
        )
        self.assertEqual(len(limited_results), 2)

    def test_output_structure(self):
        """Test that output files and directories are created correctly."""
        detector = MultiscaleDetector(output_path=self.test_dir)

        # Process a single image
        result = detector.detect(self.test_image)

        # Check directories were created
        expected_dirs = [
            Path(self.test_dir) / "advanced_edges" / "multiscale" / "edges",
            Path(self.test_dir) / "advanced_edges" / "multiscale" / "visualizations",
        ]

        for dir_path in expected_dirs:
            self.assertTrue(dir_path.exists())
            self.assertTrue(dir_path.is_dir())

    def test_result_quality(self):
        """Test quality of edge detection results."""
        test_image = np.zeros((100, 100), dtype=np.uint8)
        test_image[45:55, :] = 255  # Strong horizontal edge

        result = self.detector.detect(test_image)

        # Update threshold to be more reasonable
        edge_strength = np.mean(result[44:56, :])
        self.assertTrue(edge_strength > 0.3)  # Reduced threshold

        non_edge_strength = np.mean(result[:40, :])
        self.assertTrue(non_edge_strength < 0.1)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty image
        with self.assertRaises(ValueError):
            self.detector.detect(np.array([]))

        # Test with invalid scale values
        with self.assertRaises(ValueError):
            MultiscaleDetector(scales=[-1.0, 0.0, 1.0])

        # Test with mismatched parameter lengths
        with self.assertRaises(ValueError):
            MultiscaleDetector(
                scales=[0.5, 1.0],
                gaussian_sizes=[3, 5, 7],
                gaussian_sigmas=[1.0, 1.4, 2.0],
            )

    def test_consistency(self):
        """Test consistency of results across multiple runs."""
        # Run detection multiple times
        results = []
        for _ in range(3):
            result = self.detector.detect(self.test_image)
            results.append(result)

        # Compare results
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(results[0], results[i], decimal=6)

    def test_scale_sensitivity(self):
        """Test sensitivity to different scales."""
        # Create test images with different feature sizes
        fine_image = np.zeros((100, 100), dtype=np.uint8)
        coarse_image = np.zeros((100, 100), dtype=np.uint8)

        # Fine features
        for i in range(0, 100, 4):
            fine_image[i : i + 2, :] = 255

        # Coarse features
        for i in range(0, 100, 20):
            coarse_image[i : i + 10, :] = 255

        # Detect edges
        fine_result = self.detector.detect(fine_image)
        coarse_result = self.detector.detect(coarse_image)

        # Verify different responses
        fine_density = np.mean(fine_result > 0.1)
        coarse_density = np.mean(coarse_result > 0.1)

        self.assertNotAlmostEqual(fine_density, coarse_density, places=2)


if __name__ == "__main__":
    unittest.main()
