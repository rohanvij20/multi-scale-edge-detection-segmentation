import unittest
import numpy as np
from src.advanced_segmentation.watershed import WatershedSegmentation

class TestWatershedSegmentation(unittest.TestCase):
    def setUp(self):
        self.segmenter = WatershedSegmentation()
    
    def test_segment(self):
        # TODO: Implement tests
        pass