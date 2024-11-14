# Multi-Scale Edge Detection and Segmentation

A computer vision project implementing and comparing different edge detection algorithms and using them as a basis for image segmentation using the Berkeley Segmentation Dataset (BSDS300).

## Project Overview

This project implements basic edge detection and segmentation algorithms, focusing on:
- Multiple edge detection algorithms (Sobel, Canny)
- Basic segmentation using edge information
- Evaluation using the Berkeley Segmentation Dataset (BSDS300)
- Comparison of different edge detection approaches

## Project Structure

```
multi-scale-edge-detection-segmentation/
├── src/
│   ├── __init__.py
│   └── edge_based_segmentation.py
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
│   └── segments/
│       ├── sobel/
│       └── canny/
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
pip install opencv-python numpy
```

3. Place the BSDS300 dataset in the `data` directory, maintaining the structure shown above.

## Usage

### Basic Usage
```python
from src.edge_based_segmentation import EdgeBasedSegmentation

# Initialize segmenter with dataset path
segmenter = EdgeBasedSegmentation("data/BSDS300")

# Load and process a single image
segmenter.load_image("100075")  # Using image ID from dataset
edges = segmenter.detect_edges_canny()  # or detect_edges_sobel()
segments = segmenter.segment_image()
segmenter.save_results("results", "canny")
```

### Process Multiple Images
```python
from src.edge_based_segmentation import process_dataset_sample

# Process 5 sample images with both Sobel and Canny
process_dataset_sample(
    dataset_path="data/BSDS300",
    output_path="results",
    n_samples=5
)
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

### Results Organization
- Separate directories for edge detection and segmentation results
- Method-specific subdirectories (sobel/canny)
- Consistent naming convention for output files

## Output Structure
```
results/
├── edges/
│   ├── sobel/
│   │   └── {image_id}_edges.jpg
│   └── canny/
│       └── {image_id}_edges.jpg
└── segments/
    ├── sobel/
    │   └── {image_id}_segments.jpg
    └── canny/
        └── {image_id}_segments.jpg
```

## Team Members
- Guneet Sachdeva (Project Lead, Edge Detection Algorithms)
- Rohan Vij (Dataset Integration, Result Analysis)
- Akshay Murali (Segmentation Implementation)
- Armandeep Singh (Result Visualization)

## References
1. Canny, J. (1986). A Computational Approach to Edge Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6), 679-698.
2. Martin, D., Fowlkes, C., Tal, D., & Malik, J. (2001). A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. In Proceedings Eighth IEEE International Conference on Computer Vision.
3. Berkeley Segmentation Dataset (BSDS300): https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/