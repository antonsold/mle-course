import configparser
import os
import unittest
import pandas as pd
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from train import Model

config = configparser.ConfigParser()
config.read("config.ini")


class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        self.model = Model()

    def test_scaler(self):
        self.assertEqual(self.model.scale(), True)

    def test_log_reg(self):
        self.assertEqual(self.model.log_reg(), True)


if __name__ == "__main__":
    unittest.main()
