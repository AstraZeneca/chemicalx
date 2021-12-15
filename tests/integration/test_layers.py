import unittest

from chemicalx.models import (
    MRHGNN,
    SSIDDI,
    MatchMaker,
    MHCADDI,
    GCNBMP,
    EPGCNDS,
    DPDDI,
    DeepSynergy,
    DeepDrug,
    DeepDDS,
    DeepDDI,
    CASTER,
    DEEPCCI,
    AuDNNSynergy,
)


class TestLayers(unittest.TestCase):
    def setUp(self):
        self.x = 2

    def test_MRHGNN(self):
        model = MRHGNN(x=2)
        assert model.x == 2
