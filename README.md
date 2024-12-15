# Multi-Scale Edge Detection and Segmentation

A computer vision project implementing and comparing different edge detection algorithms and using them as a basis for image segmentation using the Berkeley Segmentation Dataset (BSDS300).

## Project Overview

This project implements comprehensive edge detection and segmentation algorithms, focusing on:
- Multiple edge detection algorithms (Sobel, Canny, LoG)
- Advanced multi-scale edge detection approach
- Basic and advanced segmentation techniques
- Evaluation using the Berkeley Segmentation Dataset (BSDS300)
- Comprehensive comparison and evaluation metrics
- Interactive segmentation interface

## Project Structure

```
multi-scale-edge-detection-segmentation/
├── src/
│   ├── advanced_edge_detection/
│   │   ├── __init__.py
│   │   ├── log_detector.py
│   │   └── multiscale_detector.py
│   ├── advanced_segmentation/
│   │   ├── __init__.py
│   │   ├── region_growing.py
│   │   └── watershed.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── ui/
│   │   ├── __init__.py
│   │   └── interactive_segmentation.py
│   ├── edge_based_segmentation.py
│   ├── clear_results.py
│   └── zip_results.py
├── data/
│   └── BSDS300/
│       ├── BSDS300-human/
│       │   └── human/
│       │       ├── color/
│       │       └── gray/
│       └── BSDS300-images/
│           ├── images/
│           │   ├── test/
│           │   └── train/
│           ├── iids_test.txt
│           └── iids_train.txt
├── results/
│   ├── edges/
│   │   ├── sobel/
│   │   └── canny/
│   ├── advanced_edges/
│   │   ├── log/
│   │   └── multiscale/
│   ├── segments/
│   │   ├── sobel/
│   │   └── canny/
│   ├── advanced_segments/
│   │   ├── watershed/
│   │   └── region_growing/
│   ├── evaluation/
│   │   ├── metrics/
│   │   └── benchmarks/
│   ├── comparisons/
│   ├── edge_detection_comparison.csv
│   └── summary_report.txt
├── tests/
│   ├── test_advanced_edge_detection/
│   │   ├── test_log_detector.py
│   │   └── test_multiscale_detector.py
│   ├── test_advanced_segmentation/
│   │   ├── test_region_growing.py
│   │   └── test_watershed.py
│   ├── test_evaluation/
│   │   └── test_metrics.py
│   └── test_edge_based_segmentation.py
├── archives/
│   └── edge_detection_results_{timestamp}.zip
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rohanvij/multi-scale-edge-detection-segmentation.git
cd multi-scale-edge-detection-segmentation
```

2. Install dependencies:
```bash
pip install opencv-python numpy pandas matplotlib scikit-learn
```

3. Place the BSDS300 dataset in the `data` directory, maintaining the structure shown above.

## Usage

### Basic Edge Detection and Segmentation
```python
from src.edge_based_segmentation import EdgeBasedSegmentation

# Initialize segmenter with dataset path
segmenter = EdgeBasedSegmentation("data/BSDS300")

# Compare edge detection methods on a single image
segmenter.load_image("100075")
metrics = segmenter.compare_edges(show_plot=True)
```

### Advanced Edge Detection
```python
from src.advanced_edge_detection.log_detector import LoGDetector
from src.advanced_edge_detection.multiscale_detector import MultiscaleDetector

# Laplacian of Gaussian detection
log_detector = LoGDetector(sigma=1.0)
log_edges = log_detector.detect(image)

# Multi-scale detection
multiscale_detector = MultiscaleDetector(scales=[0.5, 1.0, 2.0])
multiscale_edges = multiscale_detector.detect(image)
```

### Advanced Segmentation
```python
from src.advanced_segmentation.watershed import WatershedSegmentation
from src.advanced_segmentation.region_growing import RegionGrowing

# Watershed segmentation
watershed_seg = WatershedSegmentation()
watershed_result = watershed_seg.segment(image, markers)

# Region growing
region_growing = RegionGrowing(threshold=128, min_region_size=100)
regions = region_growing.segment(image)
```

### Process Multiple Images
```python
from src.edge_based_segmentation import process_dataset_sample

# Process all images with comprehensive comparison
df, summary = process_dataset_sample(
    dataset_path="data/BSDS300",
    output_path="results",
    n_samples=None,  # Process all images
    include_test=True  # Include test set
)
```

### Advanced Evaluation
```python
from src.evaluation.metrics import AdvancedMetrics

metrics = AdvancedMetrics()
iou_score = metrics.calculate_iou(predicted, ground_truth)
f1_score = metrics.calculate_boundary_f1(predicted, ground_truth)
precision, recall = metrics.calculate_precision_recall(predicted, ground_truth)
```

### Utilities
```bash
# Clear all results (with options)
python clear_results.py --results-dir custom_results --quiet

# Archive results
python zip_results.py
```

## Features

### Edge Detection
- Basic Edge Detectors
  - Sobel edge detector (adjustable kernel size)
  - Canny edge detector (adjustable thresholds)
- Advanced Edge Detectors
  - Laplacian of Gaussian (LoG)
  - Multi-scale detection approach

### Segmentation
- Basic Segmentation
  - Edge-based segmentation
  - Connected components labeling
  - Region size filtering
- Advanced Segmentation
  - Watershed segmentation
  - Region growing algorithm

### Evaluation
- Basic Metrics
  - Edge density
  - Segment count
  - Mean edge strength
- Advanced Metrics
  - Intersection over Union (IoU)
  - Boundary F1 score
  - Precision and recall

### Results Organization
```
results/
├── edges/                      # Basic edge detection results
├── advanced_edges/             # Advanced edge detection results
├── segments/                   # Basic segmentation results
├── advanced_segments/          # Advanced segmentation results
├── evaluation/                 # Evaluation metrics and benchmarks
├── comparisons/               # Visual comparisons
└── summary_report.txt         # Comprehensive analysis
```

## Team Members and Responsibilities

- Guneet Sachdeva (Project Lead)
  - Edge detection algorithms
  - LoG implementation
  - Overall architecture

- Rohan Vij
  - Dataset integration
  - Multi-scale approach
  - Results analysis

- Akshay Murali
  - Segmentation implementation
  - Watershed algorithm
  - Performance optimization

- Armandeep Singh
  - Result visualization
  - Evaluation metrics
  - Documentation

## References

1. Canny, J. (1986). A Computational Approach to Edge Detection. IEEE PAMI, 8(6), 679-698.
2. Martin, D., et al. (2001). A Database of Human Segmented Natural Images and its Application.
3. Lindeberg, T. (1998). Edge Detection and Ridge Detection with Automatic Scale Selection.
4. Berkeley Segmentation Dataset: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/