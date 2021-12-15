import unittest

from chemicalx.models import (
    CASTER,
    DPDDI,
    EPGCNDS,
    GCNBMP,
    MHCADDI,
    MRHGNN,
    SSIDDI,
    AUDNNSynergy,
    DeepCCI,
    DeepDDI,
    DeepDDS,
    DeepDrug,
    DeepSynergy,
    MatchMaker,
)


class TestLayers(unittest.TestCase):
    def setUp(self):
        self.x = 2

    def test_MRHGNN(self):
        model = MRHGNN(x=2)
        assert model.x == 2
