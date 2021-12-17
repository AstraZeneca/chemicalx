import unittest

from chemicalx.models import (
    CASTER,
    DPDDI,
    EPGCNDS,
    GCNBMP,
    MHCADDI,
    MRGNN,
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

    def test_CASTER(self):
        model = CASTER(x=2)
        assert model.x == 2

    def test_DPDDI(self):
        model = DPDDI(x=2)
        assert model.x == 2

    def test_MRGNN(self):
        model = MRGNN(x=2)
        assert model.x == 2
