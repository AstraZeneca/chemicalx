"""Tests for utils."""

import unittest

import torch

from chemicalx.utils import segment_softmax


class TestPipeline(unittest.TestCase):
    """Test the utils."""

    def test_segment_softmax(self):
        """Set up the test case with some data."""
        logit = torch.FloatTensor([-0.5, -2.5, 0.5, 1.5])
        number_of_segments = torch.LongTensor([2])
        segmentation_index = torch.LongTensor([0, 0, 1, 1])
        index = torch.LongTensor([0, 1, 2, 3])
        temperature = torch.LongTensor([2, 2, 2, 2])
        truth = torch.FloatTensor([0.7311, 0.2689, 0.3775, 0.6225])
        segment_s = segment_softmax(logit, number_of_segments, segmentation_index, index, temperature)
        difference = torch.sum(torch.abs(truth - segment_s))
        assert difference < 0.001
