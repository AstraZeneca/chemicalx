"""Example with MHCADDI."""

from chemicalx import pipeline
from chemicalx.data.datasetloader import TwoSides
from chemicalx.models.mhcaddi import MHCADDI


def main():
    """Train and evaluate the MHCADDI model."""
    dataset = TwoSides()
    model = MHCADDI(atom_feature_channels=69, atom_type_channels=100, bond_type_channels=12)

    results = pipeline(
        dataset=dataset,
        model=model,
        batch_size=2048,
        epochs=10,
        context_features=False,
        drug_features=True,
        drug_molecules=True,
    )
    results.summarize()


if __name__ == "__main__":
    main()
