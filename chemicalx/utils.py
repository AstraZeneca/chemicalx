"""Utilities for chemicalx."""

import logging

import numpy as np
import torch
from torch.types import Device

logger = logging.getLogger(__name__)


def segment_max(
    logit: torch.FloatTensor,
    number_of_segments: torch.LongTensor,
    segmentation_index: torch.LongTensor,
    index: torch.LongTensor,
):
    """Segmentation maximal index finder.

    :param logit: Logit vector.
    :param number_of_segments: Segment numbers.
    :param segmentation_index: Index of segments.
    :param index: Global index
    :returns: Largest index in each segmentation.
    """
    max_number_of_segments = index.max().item() + 1
    segmentation_max = logit.new_full((number_of_segments, max_number_of_segments), -np.inf)
    segmentation_max = segmentation_max.index_put_((segmentation_index, index), logit).max(dim=1)[0]
    return segmentation_max[segmentation_index]


def segment_sum(logit: torch.FloatTensor, number_of_segments: torch.LongTensor, segmentation_index: torch.LongTensor):
    """Segmentation sum calculation.

    :param logit: Logit vector.
    :param number_of_segments: Segment numbers.
    :param segmentation_index: Index of segments.
    :returns: Sum of logits on segments.
    """
    norm = logit.new_zeros(number_of_segments).index_add(0, segmentation_index, logit)
    return norm[segmentation_index]


def segment_softmax(
    logit: torch.FloatTensor,
    number_of_segments: torch.LongTensor,
    segmentation_index: torch.LongTensor,
    index: torch.LongTensor,
    temperature: torch.FloatTensor,
):
    """Segmentation softmax calculation.

    :param logit: Logit vector.
    :param number_of_segments: Segment numbers.
    :param segmentation_index: Index of segmentation.
    :param index: Global index.
    :param temperature: Normalization values.
    :returns: Probability scores for attention.
    """
    logit_max = segment_max(logit, number_of_segments, segmentation_index, index).detach()
    logit = torch.exp((logit - logit_max) / temperature)
    logit_norm = segment_sum(logit, number_of_segments, segmentation_index)
    prob = logit / (logit_norm + torch.finfo(logit_norm.dtype).eps)
    return prob


def resolve_device(device: Device = None) -> torch.device:
    """Resolve a :class:`torch.device` given a desired device name.

    :param device: A pre-instantiated :class:`torch.device`, a string to infer
        the device from, or none to try using GPU is possible.
    :return: A device object.

    Implementation borrowed from :func:`pykeen.utils.resolve_device`.
    """
    if device is None or device == "gpu":
        device = "cuda"
    if isinstance(device, str):
        device = torch.device(device)
    if not torch.cuda.is_available() and device.type == "cuda":
        device = torch.device("cpu")
        logger.warning("No cuda devices were available. CPU will be used.")
    return device
