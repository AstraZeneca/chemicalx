"""Example with DeepDDs."""

from chemicalx import pipeline
from chemicalx.models import DeepDDS


def main():
    """Train and evaluate the DeepDDs model."""
    model = DeepDDS(
        context_feature_size=112, context_output_size=10, in_channels=69  # context feature width; datset specific
    )
    results = pipeline(
        dataset="drugcombdb",
        model=model,
        batch_size=5120,
        epochs=1,
        context_features=True,
        drug_features=True,
        drug_molecules=True,
        labels=True,
    )
    results.summarize()


if __name__ == "__main__":
    main()
