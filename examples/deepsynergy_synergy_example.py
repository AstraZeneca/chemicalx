"""Example with DeepSynergy.

Because the OncoPolyPharmacology dataset has a continuous
output instead of a binary one, the MSE loss is used instead
of the BCE loss.
"""

from torch import nn

from chemicalx import pipeline
from chemicalx.data import OncoPolyPharmacology
from chemicalx.models import DeepSynergy


def main():
    """Train and evaluate the DeepSynergy model."""
    dataset = OncoPolyPharmacology()
    dataset.summarize()
    model = DeepSynergy(context_channels=dataset.context_channels, drug_channels=dataset.drug_channels)
    results = pipeline(
        dataset=dataset,
        model=model,
        batch_size=5120,
        epochs=100,
        context_features=True,
        drug_features=True,
        drug_molecules=False,
        loss_cls=nn.MSELoss,
        metrics=["mae", "mse"],
    )
    results.summarize()


if __name__ == "__main__":
    main()
