import unittest


class TestLayers(unittest.TestCase):
    def setUp(self):
        self.x = 2

    def test_bla(self):
        assert self.x == 2