"""Example with DeepSynergy."""

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from chemicalx.data import BatchGenerator, DatasetLoader
from chemicalx.models import DeepSynergy

loader = DatasetLoader("drugcombdb")

drug_feature_set = loader.get_drug_features()
context_feature_set = loader.get_context_features()
labeled_triples = loader.get_labeled_triples()

train_triples, test_triples = labeled_triples.train_test_split()

generator = BatchGenerator(
    batch_size=5120, context_features=True, drug_features=True, drug_molecules=False, labels=True
)

generator.set_data(context_feature_set, drug_feature_set, train_triples)

model = DeepSynergy(context_channels=112, drug_channels=256)

optimizer = torch.optim.Adam(model.parameters())

model.train()

loss = torch.nn.BCELoss()

for _ in tqdm(range(100)):
    for batch in generator:
        optimizer.zero_grad()

        prediction = model(batch.context_features, batch.drug_features_left, batch.drug_features_right)

        loss_value = loss(prediction, batch.labels)
        loss_value.backward()
        optimizer.step()

model.eval()

generator.set_labeled_triples(test_triples)

predictions = []
for batch in generator:
    prediction = model(batch.context_features, batch.drug_features_left, batch.drug_features_right)
    prediction = prediction.detach().cpu().numpy()
    identifiers = batch.identifiers
    identifiers["prediction"] = prediction
    predictions.append(identifiers)

predictions = pd.concat(predictions)
au_roc = roc_auc_score(predictions["label"], predictions["prediction"])
print(f"AUROC : {au_roc:.4f}")
