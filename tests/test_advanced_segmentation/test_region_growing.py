import unittest
import numpy as np
from src.advanced_segmentation.region_growing import RegionGrowing

class TestRegionGrowing(unittest.TestCase):
    def setUp(self):
        self.segmenter = RegionGrowing()
    
    def test_segment(self):
        # TODO: Implement tests
        pass