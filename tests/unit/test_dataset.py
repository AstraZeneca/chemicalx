import unittest

from chemicalx.dataset import (
    BaseDataset,
    ChChMinerDataset,
    DPDDIOneDataset,
    DrugCombDataset,
    DrugCombDBDataset,
    OncolyPharmDataset,
    SynergyXDBDataset,
    TwoSidesDataset,
    ZhangDDIDataset,
)


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.x = 2

    def test_BaseDataset(self):
        dataset = BaseDataset(x=2)
        assert dataset.x == 2

    def test_ChChMinerDataset(self):
        dataset = ChChMinerDataset(x=2)
        assert dataset.x == 2

    def test_DPDDIOneDataset(self):
        dataset = DPDDIOneDataset(x=2)
        assert dataset.x == 2

    def test_DrugCombDataset(self):
        dataset = DrugCombDataset(x=2)
        assert dataset.x == 2

    def test_DrugCombDBDataset(self):
        dataset = DrugCombDBDataset(x=2)
        assert dataset.x == 2

    def test_OncolyPharmDataset(self):
        dataset = OncolyPharmDataset(x=2)
        assert dataset.x == 2

    def test_SynergyXDBDataset(self):
        dataset = SynergyXDBDataset(x=2)
        assert dataset.x == 2

    def test_TwoSidesDataset(self):
        dataset = TwoSidesDataset(x=2)
        assert dataset.x == 2

    def test_ZhangDDIDataset(self):
        dataset = ZhangDDIDataset(x=2)
        assert dataset.x == 2
