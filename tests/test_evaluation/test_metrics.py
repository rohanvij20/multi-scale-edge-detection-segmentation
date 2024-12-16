import unittest
import numpy as np
from src.evaluation.metrics import SegmentationMetrics


class TestSegmentationMetrics(unittest.TestCase):
    """Test cases for segmentation metrics."""

    def setUp(self):
        """Set up test cases."""
        # Create test segmentation masks
        self.pred_mask = np.zeros((100, 100), dtype=np.uint8)
        self.pred_mask[25:75, 25:75] = 1  # Center square

        self.gt_mask = np.zeros((100, 100), dtype=np.uint8)
        self.gt_mask[20:70, 20:70] = 1  # Slightly offset square

        # Create test edges
        self.pred_edges = np.zeros((100, 100), dtype=np.uint8)
        self.pred_edges[25, 25:75] = 1  # Top edge
        self.pred_edges[75, 25:75] = 1  # Bottom edge
        self.pred_edges[25:75, 25] = 1  # Left edge
        self.pred_edges[25:75, 75] = 1  # Right edge

        self.gt_edges = np.zeros((100, 100), dtype=np.uint8)
        self.gt_edges[20, 20:70] = 1
        self.gt_edges[70, 20:70] = 1
        self.gt_edges[20:70, 20] = 1
        self.gt_edges[20:70, 70] = 1

    def test_iou(self):
        """Test IoU calculation."""
        iou = SegmentationMetrics.calculate_iou(self.pred_mask, self.gt_mask)
        self.assertTrue(0 < iou < 1)  # Should be non-zero but less than 1

        # Test perfect match
        perfect_iou = SegmentationMetrics.calculate_iou(self.pred_mask, self.pred_mask)
        self.assertEqual(perfect_iou, 1.0)

        # Test no overlap
        no_overlap_mask = np.zeros_like(self.pred_mask)
        no_overlap_mask[0:10, 0:10] = 1
        no_overlap_iou = SegmentationMetrics.calculate_iou(
            self.pred_mask, no_overlap_mask
        )
        self.assertEqual(no_overlap_iou, 0.0)

    def test_boundary_f1(self):
        """Test boundary F1 score calculation."""
        f1 = SegmentationMetrics.calculate_boundary_f1(self.pred_edges, self.gt_edges)
        self.assertTrue(0 < f1 < 1)

        # Test perfect match
        perfect_f1 = SegmentationMetrics.calculate_boundary_f1(
            self.pred_edges, self.pred_edges
        )
        self.assertAlmostEqual(perfect_f1, 1.0)

        # Test with different tolerances
        f1_tight = SegmentationMetrics.calculate_boundary_f1(
            self.pred_edges, self.gt_edges, tolerance=1
        )
        f1_loose = SegmentationMetrics.calculate_boundary_f1(
            self.pred_edges, self.gt_edges, tolerance=3
        )
        self.assertTrue(
            f1_tight < f1_loose
        )  # Looser tolerance should give higher score

    def test_segment_metrics(self):
        """Test segmentation metrics calculation."""
        # Create test segmentation with known properties
        segments = np.zeros((100, 100), dtype=np.uint8)
        segments[0:50, 0:50] = 1  # 2500 pixels
        segments[50:100, 0:50] = 2  # 2500 pixels
        segments[0:100, 50:100] = 3  # 5000 pixels

        metrics = SegmentationMetrics.segment_metrics(segments)

        self.assertEqual(metrics["num_segments"], 3)
        self.assertAlmostEqual(metrics["avg_segment_size"], 3333.33, places=2)
        self.assertTrue(metrics["min_segment_size"] == 2500)
        self.assertTrue(metrics["max_segment_size"] == 5000)

    def test_edge_metrics(self):
        """Test edge detection metrics calculation."""
        metrics = SegmentationMetrics.edge_metrics(self.pred_edges)

        self.assertTrue(0 < metrics["edge_density"] < 1)
        self.assertTrue(0 <= metrics["edge_strength_mean"] <= 1)
        self.assertTrue(metrics["edge_strength_std"] >= 0)
        self.assertTrue(metrics["edge_continuity"] > 0)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty arrays
        with self.assertRaises(Exception):
            SegmentationMetrics.calculate_iou(np.array([]), np.array([]))

        # Test with mismatched shapes
        with self.assertRaises(Exception):
            SegmentationMetrics.calculate_boundary_f1(
                np.zeros((10, 10)), np.zeros((20, 20))
            )

        # Test with invalid segments
        with self.assertRaises(Exception):
            SegmentationMetrics.segment_metrics(np.array([]))


if __name__ == "__main__":
    unittest.main()
