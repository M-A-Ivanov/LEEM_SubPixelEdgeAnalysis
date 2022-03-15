import os
import numpy as np

from edge_processing import EdgeAnalyzer

EDGE = 'edge 1'
REGION = '3microns, 0.2s, r2'
# SRC_FOLDER = os.path.join(r"C:\Users\User\OneDrive - Cardiff University\Data\LEEM\27042021\fluctuations", REGION)

SRC_FOLDER = os.path.join(r"C:\Users\c1545871\OneDrive - Cardiff University\Data\LEEM\27042021\fluctuations", REGION)
# TARGET_FOLDER = os.path.join(r"E:\results", REGION)
TARGET_FOLDER = os.path.join(r"E:\results", "testcase", REGION)
RESULTS_PATH = os.path.join(TARGET_FOLDER, EDGE, 'manual_mask')


# class TestProcess(unittest.TestCase):
#     def __init__(self):
#         super(TestProcess, self).__init__()
#         self.respath1 = os.path.join(TARGET_FOLDER, "1")
#         self.respath2 = os.path.join(TARGET_FOLDER, "2")
#
#     def create_coordinates(self):
#         num_images = 20
#         manual_masking(SRC_FOLDER, self.respath1, EDGE, num_images=num_images)
#         manual_masking(SRC_FOLDER, self.respath2, EDGE, num_images=num_images)


def test_projection_method():
    a = EdgeAnalyzer(None)
    lpt1 = np.array([0, 0])
    lpt2 = np.array([0, 2])
    test_pts = [np.array([1, 1]),
                np.array([0, 0]),
                np.array([0, 2])]
    expected = [np.array([0, 1]),
                np.array([0, 0]),
                np.array([0, 2])]
    for pt, expect in zip(test_pts, expected):
        ptonl = a.point_on_line(lpt1, lpt2, pt)
        np.testing.assert_array_equal(expect, ptonl)


def test_distance_method():
    a = EdgeAnalyzer(None)
    pt1 = np.array([0, 0])
    test_pts = [np.array([1, 1]),
                np.array([0, 0]),
                np.array([0, 2]),
                np.array([0, -2])]
    expected = [np.sqrt(2),
                0,
                2,
                2]
    for pt, expect in zip(test_pts, expected):
        dist = a.dist_to_point(pt1, pt)
        np.testing.assert_array_equal(expect, dist)

