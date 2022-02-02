"""Example with MatchMaker."""

from chemicalx import pipeline
from chemicalx.data import DrugCombDB
from chemicalx.models import MatchMaker


def main():
    """Train and evaluate the MatchMaker model."""
    dataset = DrugCombDB()
    model = MatchMaker(context_channels=dataset.context_channels, drug_channels=dataset.drug_channels)

    results = pipeline(
        dataset=dataset,
        model=model,
        batch_size=5120,
        epochs=100,
        context_features=True,
        drug_features=True,
        drug_molecules=False,
        metrics=["roc_auc"],
    )
    results.summarize()


if __name__ == "__main__":
    main()
