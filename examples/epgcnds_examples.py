"""Example with EPGCNDS."""

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from chemicalx.data import BatchGenerator, DatasetLoader
from chemicalx.models import EPGCNDS

loader = DatasetLoader("drugcombdb")

drug_feature_set = loader.get_drug_features()
context_feature_set = loader.get_context_features()
labeled_triples = loader.get_labeled_triples()


generator = BatchGenerator(batch_size=1024, context_features=True, drug_features=True, drug_molecules=True, labels=True)

train_triples, test_triples = labeled_triples.train_test_split()

generator.set_data(context_feature_set, drug_feature_set, train_triples)


model = EPGCNDS(69)

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=10 ** -7)

loss = torch.nn.BCELoss()

for epoch in range(20):
    for batch in tqdm(generator):
        optimizer.zero_grad()
        prediction = model(batch.drug_molecules_left, batch.drug_molecules_right)
        output = loss(prediction, batch.labels)
        output.backward()
        optimizer.step()

model.eval()
generator.set_labeled_triples(test_triples)

predictions = []
for batch in tqdm(generator):
    prediction = model(batch.drug_molecules_left, batch.drug_molecules_right)
    prediction = prediction.detach().cpu().numpy()
    identifiers = batch.identifiers
    identifiers["prediction"] = prediction
    predictions.append(identifiers)

predictions = pd.concat(predictions)
au_roc = roc_auc_score(predictions["label"], predictions["prediction"])
print(f"AUROC : {au_roc:.4f}")
