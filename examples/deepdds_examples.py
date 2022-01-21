"""Example with DeepDDs."""

from chemicalx import pipeline
from chemicalx.data import DrugCombDB
from chemicalx.models import DeepDDS


def main():
    """Train and evaluate the DeepDDs model."""
    dataset = DrugCombDB()
    model = DeepDDS(
        context_feature_size=dataset.context_channels,
        context_output_size=dataset.drug_channels,
        dropout=0.5,  # Rate used in source paper for DeepDDS
    )
    results = pipeline(
        dataset=dataset,
        model=model,
        batch_size=5120,
        epochs=1,
        context_features=True,
        drug_features=True,
        drug_molecules=True,
    )
    results.summarize()


if __name__ == "__main__":
    main()
