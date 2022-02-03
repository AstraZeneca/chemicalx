"""Example with DeepSynergy."""

from chemicalx import pipeline
from chemicalx.data.datasetloader import TwoSides
from chemicalx.models.mhcaddi import MHCADDI


def main():
    """Train and evaluate the DeepSynergy model."""
    dataset = TwoSides()
    model = MHCADDI(d_atom_feat=69, n_atom_type=100, n_bond_type=12)

    results = pipeline(
        dataset=dataset,
        model=model,
        batch_size=128,
        epochs=10,
        context_features=False,
        drug_features=True,
        drug_molecules=True,
    )
    results.summarize()


if __name__ == "__main__":
    main()
