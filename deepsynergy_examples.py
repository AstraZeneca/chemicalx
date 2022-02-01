"""Example with DeepSynergy."""

from chemicalx import pipeline
from chemicalx.data import DrugComb, DrugbankDDI, TwoSides
from chemicalx.models import EPGCNDS, DeepSynergy, GCNBMP
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, f1_score
import numpy as np

def main():
    """Train and evaluate the DeepSynergy model."""
    dataset = DrugComb()
    out = []
    for state in range(10):
        model = EPGCNDS()
        results = pipeline(
            dataset=dataset,
            model=model,
            batch_size=4096,
            epochs=50,
            random_state=state,
            context_features=True,
            drug_features=True,
            drug_molecules=True,
            metrics=[
                "roc_auc",
            ],
        )
        print(results.predictions.shape)
        results = results.predictions
        roc_auc = roc_auc_score(results["label"], results["prediction"])
        precision, recall, thresholds = precision_recall_curve(results["label"], results["prediction"])
        pr_auc = auc(recall, precision)
        results["label"] = results["label"].map(lambda x: int(x))
        results["prediction"] = results["prediction"].map(lambda x: 1 if x>0.5 else 0)
        f_1_value = f1_score(results["label"], results["prediction"])
        print([roc_auc,pr_auc,f_1_value])
        out.append([roc_auc,pr_auc,f_1_value])
    out = np.array(out)
    print(np.mean(out, axis=0))
    print(np.std(out, axis=0)/(10**0.5))


if __name__ == "__main__":
    main()
