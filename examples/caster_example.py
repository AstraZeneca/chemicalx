"""Example with CASTER."""

from chemicalx import pipeline
from chemicalx.data import DrugCombDB
from chemicalx.loss import CASTERSupervisedLoss
from chemicalx.models import CASTER


def main():
    """Train and evaluate the CASTER model."""
    dataset = DrugCombDB()
    model = CASTER(drug_channels=dataset.drug_channels)
    results = pipeline(
        dataset=dataset,
        model=model,
        loss_cls=CASTERSupervisedLoss,
        batch_size=5120,
        epochs=1,
        context_features=False,
        drug_features=True,
        drug_molecules=False,
        metrics=[
            "roc_auc",
        ],
    )
    results.summarize()


if __name__ == "__main__":
    main()
