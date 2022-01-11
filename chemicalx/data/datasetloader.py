import io
import json
import numpy as np
import pandas as pd
import urllib.request
from typing import Dict
from chemicalx.data import DrugFeatureSet, ContextFeatureSet, LabeledTriples


class DatasetLoader:
    def __init__(self, dataset_name: str):
        self.base_url = "https://raw.githubusercontent.com/AstraZeneca/chemicalx/main/dataset"
        self.dataset_name = dataset_name
        assert dataset_name in ["drugcombdb", "drugcomb"]

    def generate_path(self, file_name: str) -> str:
        data_path = "/".join([self.base_url, self.dataset_name, file_name])
        return data_path

    def load_raw_json_data(self, path: str) -> Dict:
        with urllib.request.urlopen(path) as url:
            raw_data = json.loads(url.read().decode())
        return raw_data

    def load_raw_csv_data(self, path: str) -> pd.DataFrame:
        data_bytes = urllib.request.urlopen(path).read()
        types = {"drug_1": str, "drug_2": str, "context": str, "label": float}
        raw_data = pd.read_csv(io.BytesIO(data_bytes), encoding="utf8", sep=",", dtype=types)
        return raw_data

    def get_context_features(self):
        path = self.generate_path("context_set.json")
        raw_data = self.load_raw_json_data(path)
        raw_data = {k: np.array(v) for k, v in raw_data.items()}
        context_feature_set = ContextFeatureSet()
        context_feature_set.update(raw_data)
        return context_feature_set

    def get_drug_features(self):
        path = self.generate_path("drug_set.json")
        raw_data = self.load_raw_json_data(path)
        raw_data = {k: {"smiles": v["smiles"], "features": np.array(v["features"])} for k, v in raw_data.items()}
        drug_feature_set = DrugFeatureSet()
        drug_feature_set.update(raw_data)
        return drug_feature_set

    def get_labeled_triples(self):
        path = self.generate_path("labeled_triples.csv")
        raw_data = self.load_raw_csv_data(path)
        labeled_triple_set = LabeledTriples()
        labeled_triple_set.update_from_pandas(raw_data)
        return labeled_triple_set
