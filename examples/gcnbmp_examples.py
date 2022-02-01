"""Example with EPGCNDS."""

from chemicalx import pipeline
from chemicalx.data import DrugCombDB
from chemicalx.models import GCNBMP


def main():
    """Train and evaluate the EPGCNDS model."""
    dataset = DrugCombDB()
    model = GCNBMP(hidden_conv_layers=1)

    print(model)

    results = pipeline(
        dataset=dataset,
        model=model,
        optimizer_kwargs=dict(lr=0.01, weight_decay=10 ** -7),
        batch_size=5,
        epochs=1,
        context_features=True,
        drug_features=True,
        drug_molecules=True,
    )
    results.summarize()


if __name__ == "__main__":
    main()
