# Multi-Scale Edge Detection and Segmentation

A computer vision project implementing and comparing different edge detection algorithms and using them as a basis for image segmentation using the Berkeley Segmentation Dataset (BSDS300).

## Project Overview

This project implements edge detection and segmentation algorithms, focusing on:
- Multiple edge detection algorithms (Sobel, Canny)
- Advanced multi-scale edge detection approach
- Basic segmentation using edge information
- Evaluation using the Berkeley Segmentation Dataset (BSDS300)
- Comprehensive comparison of different edge detection approaches
- Visualization and metrics for algorithm comparison

## Project Structure

```
multi-scale-edge-detection-segmentation/
├── src/
│   ├── advanced_edge_detection/
│   │   ├── __init__.py
│   │   └── multiscale_detector.py        # Multi-scale implementation
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
├── tests/
│   ├── test_advanced_edge_detection/
│   │   └── test_multiscale_detector.py   # Multi-scale tests
│   └── test_edge_based_segmentation.py
├── results/
│   ├── edges/
│   │   ├── sobel/
│   │   └── canny/
│   ├── advanced_edges/
│   │   └── multiscale/
│   │       ├── edges/          # Multi-scale edge detection results
│   │       └── visualizations/ # Multi-scale visualizations
│   ├── segments/
│   │   ├── sobel/
│   │   └── canny/
│   ├── comparisons/
│   ├── edge_detection_comparison.csv
│   └── summary_report.txt
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

3. Set up Python path:
```bash
export PYTHONPATH="/path/to/multi-scale-edge-detection-segmentation:$PYTHONPATH"
```

4. Place the BSDS300 dataset in the `data` directory.

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

### Multi-scale Edge Detection
```python
from src.advanced_edge_detection import MultiscaleDetector

# Initialize detector with custom parameters
detector = MultiscaleDetector(
    scales=[0.5, 1.0, 2.0],
    gaussian_sizes=[3, 5, 7],
    gaussian_sigmas=[1.0, 1.4, 2.0]
)

# Process single image
result = detector.detect(image, method='weighted')

# Process entire dataset
results_df = detector.process_dataset(
    "data/BSDS300",
    subset='all',      # Process both train and test sets
    n_samples=None     # Process all images
)
```

### Process Multiple Images with Comparison
```python
from src.edge_based_segmentation import process_dataset_sample

# Process dataset with comprehensive comparison
df, summary = process_dataset_sample(
    dataset_path="data/BSDS300",
    output_path="results",
    n_samples=None,    # Process all images
    include_test=True  # Include test set
)
```

### Utilities
```bash
# Clear all results
python clear_results.py

# Archive results
python zip_results.py
```

## Features

### Edge Detection
1. Basic Edge Detectors:
   - Sobel edge detector
     - Adjustable kernel size (3, 5, 7)
     - Gradient magnitude computation
     - Automatic normalization
   - Canny edge detector
     - Adjustable hysteresis thresholds
     - Built-in noise reduction
     - Non-maximum suppression

2. Multi-scale Edge Detection:
   - Multiple scale processing (0.5x, 1.0x, 2.0x)
   - Scale-specific Gaussian smoothing
   - Weighted combination of scales
   - Automatic scale normalization
   - Comprehensive edge strength analysis

### Segmentation
- Edge-based region growing
- Minimum region size filtering
- Connected components labeling

### Comparison and Analysis
- Side-by-side visualization of methods
- Quantitative metrics:
  - Edge density
  - Number of segments
  - Mean edge strength
  - Scale-specific metrics
- Statistical summaries
- CSV reports for detailed analysis

### Results Organization
```
results/
├── edges/                      # Basic edge detection results
│   ├── sobel/
│   └── canny/
├── advanced_edges/
│   └── multiscale/            # Multi-scale results
│       ├── edges/             # Edge detection outputs
│       └── visualizations/    # Multi-scale visualizations
├── segments/                  # Segmentation results
├── comparisons/              # Method comparisons
├── edge_detection_comparison.csv
└── summary_report.txt
```

### Utilities
- `clear_results.py`: Clear all results while maintaining directory structure
- `zip_results.py`: Create timestamped archives of results

## Team Members
- Guneet Sachdeva (Project Lead, Edge Detection Algorithms)
- Rohan Vij (Multi-scale Edge Detection, Dataset Integration)
- Akshay Murali (Segmentation Implementation)
- Armandeep Singh (Result Visualization)

## References
1. Canny, J. (1986). A Computational Approach to Edge Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6), 679-698.
2. Martin, D., Fowlkes, C., Tal, D., & Malik, J. (2001). A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics.
3. Lindeberg, T. (1998). Edge detection and ridge detection with automatic scale selection. International Journal of Computer Vision, 30(2), 117-154.
4. Berkeley Segmentation Dataset (BSDS300): https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/