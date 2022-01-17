import json
import random
import rdkit
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

from tdc.multi_pred import DDI

DDI(name="DrugBank")

positive_samples = pd.read_csv("./data/drugbank.tab", sep="\t")

print(positive_samples.shape)

drugs_raw = {}
big_map = {}

for sample in positive_samples.values.tolist():
    drugs_raw[sample[0]] = sample[-2]
    drugs_raw[sample[1]] = sample[-1]
    big_map[(sample[0], sample[1], sample[2])] = 1
    big_map[(sample[1], sample[0], sample[2])] = 1

drugs = list(drugs_raw.keys())
contexts = list(set(positive_samples["Y"].values.tolist()))

print(len(drugs))
print(len(contexts))

negative_samples = []

labeled_triples = positive_samples[["ID1", "ID2", "Y"]]
labeled_triples.columns = ["drug_1", "drug_2", "context"]
labeled_triples["label"] = 1.0

for _ in tqdm(range(len(positive_samples.values.tolist()))):
    drug_1, drug_2 = random.sample(drugs, 2)
    context = random.choice(contexts)
    if (drug_1, drug_2, context) in big_map:
        while (drug_1, drug_2, context) in big_map:
            drug_1, drug_2 = random.sample(drugs, 2)
            context = random.choice(contexts)
    negative_sample = [drug_1, drug_2, context, 0.0]
    negative_samples.append(negative_sample)

negative_samples = pd.DataFrame(negative_samples, columns=["drug_1", "drug_2", "context", "label"])

labeled_triples = pd.concat([labeled_triples, negative_samples])
labeled_triples["context"] = labeled_triples["context"].map(lambda x: "context_" + str(x))

print(labeled_triples.shape)

labeled_triples.to_csv("labeled_triples.csv", index=None)

drugs_raw[
    "DB09323"
] = "O.O.O.O.C(CNCC1=CC=CC=C1)NCC1=CC=CC=C1.[H][C@]12SC(C)(C)[C@@H](N1C(=O)[C@H]2NC(=O)CC1=CC=CC=C1)C(O)=O.[H][C@]12SC(C)(C)[C@@H](N1C(=O)[C@H]2NC(=O)CC1=CC=CC=C1)C(O)=O"
drugs_raw[
    "DB13450"
] = "[O-]S(=O)(=O)C1=CC=CC=C1.[O-]S(=O)(=O)C1=CC=CC=C1.COC1=CC2=C(C=C1OC)[C@@H](CC1=CC(OC)=C(OC)C=C1)[N@@+](C)(CCC(=O)OCCCCCOC(=O)CC[N@@+]1(C)CCC3=C(C=C(OC)C(OC)=C3)[C@H]1CC1=CC(OC)=C(OC)C=C1)CC2"
drugs_raw["DB09396"] = "O.OS(=O)(=O)C1=CC2=CC=CC=C2C=C1.CCC(=O)O[C@@](CC1=CC=CC=C1)([C@H](C)CN(C)C)C1=CC=CC=C1"
drugs_raw["DB09162"] = "[Fe+3].OC(CC([O-])=O)(CC([O-])=O)C([O-])=O"
drugs_raw["DB11106"] = "CC(C)(N)CO.CN1C2=C(NC(Br)=N2)C(=O)N(C)C1=O"
drugs_raw[
    "DB11630"
] = "C1CC2=NC1=C(C3=CC=C(N3)C(=C4C=CC(=N4)C(=C5C=CC(=C2C6=CC(=CC=C6)O)N5)C7=CC(=CC=C7)O)C8=CC(=CC=C8)O)C9=CC(=CC=C9)O"
drugs_raw["DB00958"] = "C1CC(C1)(C(=O)O)C(=O)O.[NH2-].[NH2-].[Pt+2]"
drugs_raw["DB00526"] = "C1CCC(C(C1)[NH-])[NH-].C(=O)(C(=O)O)O.[Pt+2]"
drugs_raw["DB13145"] = "C(C(=O)O)O.[NH2-].[NH2-].[Pt+2]"
drugs_raw["DB00515"] = "N.N.Cl[Pt]Cl"

drug_set = {}
for drug, smiles in drugs_raw.items():
    drug_set[drug] = {}
    drug_set[drug]["smiles"] = smiles
    molecule = rdkit.Chem.MolFromSmiles(smiles)
    features = AllChem.GetHashedMorganFingerprint(molecule, 2, nBits=256)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(features, array)
    drug_features = array.tolist()
    drug_set[drug]["features"] = drug_features

with open("drug_set.json", "w") as f:
    json.dump(drug_set, f)

context_count = len(contexts)


def map_context(index, countext_count):
    context_vector = [0 for i in range(countext_count)]
    context_vector[index] = 1
    return context_vector


context_set = {context: map_context(i, context_count) for i, context in enumerate(contexts)}

with open("context_set.json", "w") as f:
    json.dump(context_set, f)
