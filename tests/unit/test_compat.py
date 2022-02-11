"""Tests for compatibility layer."""

import unittest

import torch.cuda
from torchdrug.data import Molecule

from chemicalx.compat import Graph, PackedGraph


@unittest.skipUnless(torch.cuda.is_available(), "can not test compatibility layer without a GPU available")
class TestCompat(unittest.TestCase):
    """Tests for the compatibility layer."""

    def setUp(self) -> None:
        """Set up the test case."""
        self.gpu_device = torch.device("cuda")

    def test_packed_graph_device(self):
        """Test the :func:`PackedGraph.to()`."""
        structures = ["C(=O)O", "CCO"]
        molecules = [Molecule.from_smiles(smiles) for smiles in structures]
        packed_graph = Graph.pack(molecules)
        self.assertIsInstance(packed_graph, PackedGraph)

        self.assertEqual("cpu", packed_graph.edge_list.device.type)
        self.assertEqual(
            -1, packed_graph.edge_list.get_device(), msg="Device should be -1 to represent it's on the CPU"
        )

        gpu_packed_graph = packed_graph.to(self.gpu_device)
        self.assertEqual("cuda", gpu_packed_graph.edge_list.device.type)
        self.assertLessEqual(
            0, gpu_packed_graph.edge_list.get_device(), msg="Device should 0 or higher for a cuda device"
        )

        self.assertIsNot(
            packed_graph,
            gpu_packed_graph,
            msg="The PackedGraph.cuda() creates a new object. There is no in-place versions of this, unfortunately",
        )
