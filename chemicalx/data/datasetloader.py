import json
import numpy as np
import pandas as pd
import urllib.request
from typing import Dict
from chemicalx.data import DrugFeatureSet, ContextFeatureSet


class DatasetLoader:
    def __init__(self, dataset_name: str):
        self.base_url = "https://raw.githubusercontent.com/AstraZeneca/chemicalx/main/dataset"
        self.dataset_name = dataset_name
        assert dataset_name in ["drugcombdb", "drugcomb"]

    def load_raw_data(self, data_part_name: str) -> Dict:
        data_path = "/".join([self.base_url, self.dataset_name, data_part_name])
        with urllib.request.urlopen(data_path) as url:
            raw_data = json.loads(url.read().decode())
        return raw_data

    def get_context_features(self):
        raw_data = self.load_raw_data("context_set.json")
        raw_data = {k: np.array(v) for k, v in raw_data.items()}
        context_feature_set = ContextFeatureSet()
        context_feature_set.update(raw_data)
        return context_feature_set

    def get_drug_features(self):
        raw_data = self.load_raw_data("drug_set.json")
        raw_data = {k: {"smiles": v["smiles"], "features": np.array(v["features"])} for k, v in raw_data.items()}
        drug_feature_set = DrugFeatureSet()
        drug_feature_set.update(raw_data)
        return drug_feature_set
