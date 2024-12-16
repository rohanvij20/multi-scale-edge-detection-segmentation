import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


class SegmentationMetrics:
    """Evaluation metrics for image segmentation."""

    @staticmethod
    def calculate_iou(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate Intersection over Union."""
        intersection = np.logical_and(predicted, ground_truth)
        union = np.logical_or(predicted, ground_truth)
        if union.sum() == 0:
            return 0
        return intersection.sum() / union.sum()

    @staticmethod
    def calculate_boundary_f1(
        predicted: np.ndarray, ground_truth: np.ndarray, tolerance: int = 2
    ) -> float:
        """Calculate Boundary F1 score."""
        # Convert to binary edge maps if needed
        pred_edges = predicted > 0
        gt_edges = ground_truth > 0

        # Calculate precision and recall with tolerance
        matched_pred = np.zeros_like(pred_edges)
        matched_gt = np.zeros_like(gt_edges)

        # Dilate ground truth and predicted edges for tolerance
        kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
        gt_dilated = cv2.dilate(gt_edges.astype(np.uint8), kernel) > 0
        pred_dilated = cv2.dilate(pred_edges.astype(np.uint8), kernel) > 0

        # Calculate matches
        matched_pred[pred_edges] = gt_dilated[pred_edges]
        matched_gt[gt_edges] = pred_dilated[gt_edges]

        # Calculate precision and recall
        precision = np.sum(matched_pred) / (np.sum(pred_edges) + 1e-7)
        recall = np.sum(matched_gt) / (np.sum(gt_edges) + 1e-7)

        # Calculate F1
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return f1

    @staticmethod
    def segment_metrics(segments: np.ndarray) -> dict:
        """Calculate metrics for segmentation result."""
        return {
            "num_segments": len(np.unique(segments)),
            "avg_segment_size": np.mean(
                [np.sum(segments == label) for label in np.unique(segments)]
            ),
            "std_segment_size": np.std(
                [np.sum(segments == label) for label in np.unique(segments)]
            ),
            "min_segment_size": np.min(
                [np.sum(segments == label) for label in np.unique(segments)]
            ),
            "max_segment_size": np.max(
                [np.sum(segments == label) for label in np.unique(segments)]
            ),
        }

    @staticmethod
    def edge_metrics(edges: np.ndarray) -> dict:
        """Calculate metrics for edge detection result."""
        return {
            "edge_density": np.mean(edges > 0),
            "edge_strength_mean": np.mean(edges),
            "edge_strength_std": np.std(edges),
            "edge_continuity": np.sum(edges > 0) / (edges.shape[0] + edges.shape[1]),
        }
