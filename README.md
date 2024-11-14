# Multi-Scale Edge Detection and Segmentation

A computer vision project implementing and comparing different edge detection algorithms and using them as a basis for image segmentation using the Berkeley Segmentation Dataset (BSDS300).

## Project Overview

This project implements basic edge detection and segmentation algorithms, focusing on:
- Multiple edge detection algorithms (Sobel, Canny)
- Basic segmentation using edge information
- Evaluation using the Berkeley Segmentation Dataset (BSDS300)
- Comprehensive comparison of different edge detection approaches
- Visualization and metrics for algorithm comparison

## Project Structure

```
multi-scale-edge-detection-segmentation/
├── src/
│   ├── __init__.py
│   ├── edge_based_segmentation.py
│   ├── clear_results.py
│   └── zip_results.py
├── data/
│   └── BSDS300/
│       ├── BSDS300-human/
│       │   └── human/
│       │       ├── color/
│       │       │   └── [1102-1132]/
│       │       └── gray/
│       │           └── [1102-1132]/
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

3. Place the BSDS300 dataset in the `data` directory, maintaining the structure shown above.

## Usage

### Basic Usage
```python
from src.edge_based_segmentation import EdgeBasedSegmentation

# Initialize segmenter with dataset path
segmenter = EdgeBasedSegmentation("data/BSDS300")

# Compare edge detection methods on a single image
segmenter.load_image("100075")
metrics = segmenter.compare_edges(show_plot=True)
```

### Process Multiple Images
```python
from src.edge_based_segmentation import process_dataset_sample

# Process 5 sample images with comprehensive comparison
df, summary = process_dataset_sample(
    dataset_path="data/BSDS300",
    output_path="results",
    n_samples=5
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
- Sobel edge detector
  - Adjustable kernel size (3, 5, 7)
  - Gradient magnitude computation
  - Automatic normalization
- Canny edge detector
  - Adjustable hysteresis thresholds
  - Built-in noise reduction
  - Non-maximum suppression

### Segmentation
- Edge-based region growing
- Watershed segmentation approach
- Minimum region size filtering
- Connected components labeling

### Comparison and Analysis
- Side-by-side visualization of methods
- Quantitative metrics:
  - Edge density
  - Number of segments
  - Mean edge strength
- Statistical summaries
- CSV reports for detailed analysis

### Results Organization
```
results/
├── edges/
│   ├── sobel/
│   │   └── {image_id}_edges.jpg
│   └── canny/
│       └── {image_id}_edges.jpg
├── segments/
│   ├── sobel/
│   │   └── {image_id}_segments.jpg
│   └── canny/
│       └── {image_id}_segments.jpg
├── comparisons/
│   └── comparison_{image_id}.png
├── edge_detection_comparison.csv
└── summary_report.txt
```

### Utilities
- `clear_results.py`: Clear all results while maintaining directory structure
- `zip_results.py`: Create timestamped archives of results

## Team Members
- Guneet Sachdeva (Project Lead, Edge Detection Algorithms)
- Rohan Vij (Dataset Integration, Result Analysis)
- Akshay Murali (Segmentation Implementation)
- Armandeep Singh (Result Visualization)

## References
1. Canny, J. (1986). A Computational Approach to Edge Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6), 679-698.
2. Martin, D., Fowlkes, C., Tal, D., & Malik, J. (2001). A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics.
3. Berkeley Segmentation Dataset (BSDS300): https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/