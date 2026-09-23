import numpy as np

from spatialcyte.segment import ThresholdSegmenter, segment_regions


def test_threshold_segmenter_finds_two_blobs():
    img = np.zeros((40, 40), np.float32)
    img[5:11, 5:11] = 255
    img[25:31, 25:31] = 255
    labels = ThresholdSegmenter(min_size_px=4).segment(img)
    assert labels.max() == 2
    assert set(np.unique(labels)) == {0, 1, 2}


def test_threshold_segmenter_blank_image():
    labels = ThresholdSegmenter().segment(np.zeros((10, 10), np.float32))
    assert labels.max() == 0


def test_segment_regions_finds_a_vessel():
    img = np.zeros((50, 50), np.float32)
    img[10:20, 10:40] = 300  # a horizontal "vessel"
    labels = segment_regions(img, min_size_px=20)
    assert labels.max() >= 1
