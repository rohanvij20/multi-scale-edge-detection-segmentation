# Multi-Scale Edge Detection and Segmentation

A computer vision project implementing and comparing different edge detection algorithms and using them as a basis for image segmentation.

## Project Overview

This project implements basic edge detection and segmentation algorithms, focusing on:
- Multiple edge detection algorithms (Sobel, Canny)
- Basic segmentation using edge information
- Evaluation using the Berkeley Segmentation Dataset (BSDS300)

## Project Structure

```
computer_vision_project/
├── src/
│   ├── __init__.py
│   └── edge_based_segmentation.py
├── data/
│   └── BSDS300/
│       ├── BSDS300-human/
│       └── BSDS300-images/
├── tests/
│   └── test_edge_based_segmentation.py
├── results/
│   ├── edges/
│   └── segments/
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi-scale-edge-detection-segmentation.git
cd multi-scale-edge-detection-segmentation
```

2. Install dependencies:
```bash
pip install opencv-python numpy
```

3. Download the BSDS300 dataset and place it in the `data` directory.

## Usage

Basic usage example:

```python
from src.edge_based_segmentation import EdgeBasedSegmentation

# Initialize segmenter
segmenter = EdgeBasedSegmentation("data/BSDS300")

# Load and process an image
segmenter.load_bsds_image("100075")  # Using image ID from dataset
edges = segmenter.detect_edges_canny()  # or detect_edges_sobel()
segments = segmenter.segment_image()
edge_vis, segment_vis = segmenter.visualize_results("results/output")
```

## Features

### Edge Detection
- Sobel edge detector with adjustable kernel size
- Canny edge detector with adjustable thresholds

### Segmentation
- Basic region-growing approach using edge information
- Minimum region size filtering
- Watershed-based segmentation

### Visualization
- Edge detection visualization
- Segmentation results visualization
- Option to save results to files

## Team Members
- Guneet Sachdeva (Project Lead)
- Rohan Vij
- Akshay Murali
- Armandeep Singh

## References
1. Canny, J. (1986). A Computational Approach to Edge Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence
2. Martin, D., Fowlkes, C., Tal, D., & Malik, J. (2001). A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics