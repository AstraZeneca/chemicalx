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


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.x = 2

    def test_CASTER(self):
        model = CASTER(x=2)
        assert model.x == 2

    def test_DPDDI(self):
        model = DPDDI(x=2)
        assert model.x == 2

    def test_EPGCNDS(self):
        model = EPGCNDS(x=2)
        assert model.x == 2

    def test_GCNBMP(self):
        model = GCNBMP(x=2)
        assert model.x == 2

    def test_MHCADDI(self):
        model = MHCADDI(x=2)
        assert model.x == 2

    def test_MRGNN(self):
        model = MRGNN(x=2)
        assert model.x == 2

    def test_SSIDDI(self):
        model = SSIDDI(x=2)
        assert model.x == 2

    def test_AUDNNSynergy(self):
        model = AUDNNSynergy(x=2)
        assert model.x == 2

    def test_DeepCCI(self):
        model = DeepCCI(x=2)
        assert model.x == 2

    def test_DeepDDI(self):
        model = DeepDDI(x=2)
        assert model.x == 2

    def test_DeepDDS(self):
        model = DeepDDS(x=2)
        assert model.x == 2

    def test_DeepDrug(self):
        model = DeepDrug(x=2)
        assert model.x == 2

    def test_DeepSynergy(self):
        model = DeepSynergy(x=2)
        assert model.x == 2

    def test_MatchMaker(self):
        model = MatchMaker(x=2)
        assert model.x == 2
