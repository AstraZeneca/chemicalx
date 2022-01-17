import json
import random
import rdkit
import heapq
import numpy as np
import pandas as pd  
from tqdm import tqdm

from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

from tdc.multi_pred import DDI

from collections import Counter

DDI(name = "TWOSIDES")

positive_samples = pd.read_csv("./data/twosides.csv", sep=",")

print(positive_samples)

context_counts = Counter(positive_samples["Side Effect Name"].values.tolist())

print(context_counts)

contexts = heapq.nlargest(10, context_counts , key=context_counts.get)

print(contexts)

positive_samples  = positive_samples[positive_samples["Side Effect Name"].isin(contexts)]

print(positive_samples.shape)

drugs_raw = {}
big_map = {}

for sample in positive_samples.values.tolist():
    drugs_raw[sample[0]] = sample[-2]
    drugs_raw[sample[1]] = sample[-1]
    big_map[(sample[0], sample[1], sample[3])] = 1
    big_map[(sample[1], sample[0], sample[3])] = 1

drugs = list(drugs_raw.keys())

print(len(drugs))
print(len(contexts))

negative_samples = []

labeled_triples = positive_samples[["ID1", "ID2", "Y"]]
labeled_triples.columns = ["drug_1","drug_2","context"]
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

negative_samples = pd.DataFrame(negative_samples, columns = ["drug_1","drug_2","context", "label"])

labeled_triples = pd.concat([labeled_triples, negative_samples])

print(labeled_triples.shape)

labeled_triples.to_csv("labeled_triples.csv", index=None)

drug_set = {}
for drug, smiles in drugs_raw.items():
    drug_set[drug] ={}
    drug_set[drug]["smiles"] = smiles
    molecule = rdkit.Chem.MolFromSmiles(smiles)
    features = AllChem.GetHashedMorganFingerprint(molecule, 2, nBits=256)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(features, array)
    drug_features = array.tolist()
    drug_set[drug]["features"] = drug_features

with open("drug_set.json","w") as f:
    json.dump(drug_set, f)

context_count = len(contexts)

def map_context(index, countext_count):
    context_vector = [0 for i in range(countext_count)]
    context_vector[index] =  1
    return context_vector

context_set = {context: map_context(i, context_count) for i, context in enumerate(contexts)}

with open("context_set.json","w") as f:
    json.dump(context_set, f)