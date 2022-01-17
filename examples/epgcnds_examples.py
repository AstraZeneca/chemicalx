"""Example with EPGCNDS."""

from chemicalx import pipeline
from chemicalx.models import EPGCNDS


def main():
    """Train and evaluate the EPGCNDS model."""
    model = EPGCNDS(in_channels=69)
    results = pipeline(
        dataset="drugcombdb",
        model=model,
        optimizer_kwargs=dict(lr=0.01, weight_decay=10 ** -7),
        batch_size=1024,
        epochs=20,
        context_features=True,
        drug_features=True,
        drug_molecules=True,
        labels=True,
    )
    results.summarize()


if __name__ == "__main__":
    main()
