"""Example with DeepDrug."""

from chemicalx import pipeline
from chemicalx.data import DrugCombDB
from chemicalx.models import DeepDrug


def main():
    """Train and evaluate the EPGCNDS model."""
    dataset = DrugCombDB()
    model = DeepDrug()
    results = pipeline(
        dataset=dataset,
        model=model,
        optimizer_kwargs=dict(lr=0.001),
        batch_size=1024,
        epochs=20,
        context_features=False,
        drug_features=True,
        drug_molecules=True,
    )
    results.summarize()


if __name__ == "__main__":
    main()
