import unittest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil
from src.advanced_segmentation import RegionGrowing


class TestRegionGrowing(unittest.TestCase):
    """Test cases for RegionGrowing segmentation."""

    def setUp(self):
        """Set up test cases."""
        self.test_dir = tempfile.mkdtemp()
        self.segmenter = RegionGrowing(output_path=self.test_dir)

        # Create test image with distinct regions
        self.test_image = np.zeros((100, 100), dtype=np.uint8)
        self.test_image[20:40, 20:40] = 100  # Top left square
        self.test_image[60:80, 60:80] = 200  # Bottom right square

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test segmenter initialization."""
        segmenter = RegionGrowing(threshold=0.1, num_seeds=50)
        self.assertEqual(segmenter.threshold, 0.1)
        self.assertEqual(segmenter.num_seeds, 50)

    def test_basic_segmentation(self):
        """Test basic segmentation without edge map."""
        segments = self.segmenter.segment(self.test_image)

        # Check output properties
        self.assertEqual(segments.shape, self.test_image.shape)
        self.assertTrue(np.any(segments > 0))

        # Check number of segments
        unique_segments = len(np.unique(segments))
        self.assertTrue(0 < unique_segments <= self.segmenter.num_seeds)

    def test_edge_guided_segmentation(self):
        """Test segmentation with edge map guidance."""
        # Create simple edge map
        edge_map = np.zeros_like(self.test_image)
        edge_map[40:60, :] = 255  # Horizontal line
        edge_map[:, 40:60] = 255  # Vertical line

        segments = self.segmenter.segment(self.test_image, edge_map)

        # Check that segmentation respects edges
        self.assertTrue(np.any(segments[0:40, 0:40] != segments[60:100, 60:100]))

    def test_minimum_region_size(self):
        """Test minimum region size filtering."""
        segmenter = RegionGrowing(threshold=0.05, num_seeds=50, min_region_size=500)

        segments = segmenter.segment(self.test_image)

        # Check that small regions are removed
        for label in np.unique(segments):
            if label == 0:  # Skip background
                continue
            region_size = np.sum(segments == label)
            self.assertTrue(region_size >= segmenter.min_region_size)

    def test_reproducibility(self):
        """Test that segmentation is reproducible with same seed."""
        # Run segmentation twice with same parameters
        result1 = self.segmenter.segment(self.test_image)
        result2 = self.segmenter.segment(self.test_image)

        np.testing.assert_array_equal(result1, result2)

    def test_color_image_handling(self):
        """Test handling of color images."""
        # Create color test image
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        color_image[20:40, 20:40] = [100, 0, 0]  # Red square
        color_image[60:80, 60:80] = [0, 100, 0]  # Green square

        segments = self.segmenter.segment(color_image)
        self.assertEqual(segments.shape, (100, 100))

    def test_output_saving(self):
        """Test saving of results."""
        image_id = "test_image"
        self.segmenter.segment(self.test_image, image_id=image_id)

        # Check if files were created
        segment_path = (
            Path(self.test_dir)
            / "advanced_segments"
            / "region_growing"
            / "segments"
            / f"{image_id}_segments.png"
        )
        vis_path = (
            Path(self.test_dir)
            / "advanced_segments"
            / "region_growing"
            / "visualizations"
            / f"{image_id}_visualization.png"
        )

        self.assertTrue(segment_path.exists())
        self.assertTrue(vis_path.exists())

    def test_process_dataset(self):
        """Test processing of multiple images."""
        # Create mock dataset
        dataset_dir = Path(self.test_dir) / "mock_dataset"
        dataset_dir.mkdir(parents=True)

        # Save test images
        for i in range(3):
            cv2.imwrite(str(dataset_dir / f"test_{i}.jpg"), self.test_image)

        # Process dataset
        results = self.segmenter.process_dataset(str(dataset_dir), n_samples=2)

        # Check output directory structure
        output_dirs = [
            Path(self.test_dir) / "advanced_segments" / "region_growing" / "segments",
            Path(self.test_dir)
            / "advanced_segments"
            / "region_growing"
            / "visualizations",
        ]
        for dir_path in output_dirs:
            self.assertTrue(dir_path.exists())

    def test_edge_map_threshold_adjustment(self):
        """Test threshold adjustment based on edge map."""
        # Create edge map with known mean value
        edge_map = np.ones_like(self.test_image) * 100

        # Store original threshold
        original_threshold = self.segmenter.threshold

        # Segment with edge map
        self.segmenter.segment(self.test_image, edge_map)

        # Check that threshold was adjusted
        self.assertNotEqual(self.segmenter.threshold, original_threshold)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty image
        with self.assertRaises(Exception):
            self.segmenter.segment(np.array([]))

        # Test with invalid parameters
        with self.assertRaises(Exception):
            RegionGrowing(threshold=-1)

        with self.assertRaises(Exception):
            RegionGrowing(num_seeds=0)


if __name__ == "__main__":
    unittest.main()
