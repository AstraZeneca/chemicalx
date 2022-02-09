"""Example with MRGNN."""

from chemicalx import pipeline
from chemicalx.data import DrugCombDB
from chemicalx.models import MRGNN


def main():
    """Train and evaluate the MRGNN model."""
    dataset = DrugCombDB()
    model = MRGNN()
    results = pipeline(
        dataset=dataset,
        model=model,
        optimizer_kwargs=dict(lr=0.01, weight_decay=10**-7),
        batch_size=1024,
        epochs=1,
        context_features=True,
        drug_features=True,
        drug_molecules=True,
    )
    results.summarize()


if __name__ == "__main__":
    main()
