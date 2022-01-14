from chemicalx import pipeline
from chemicalx.models import DeepSynergy


def main():
    """Train the DeepSynergy model."""
    model = DeepSynergy(context_channels=112, drug_channels=256)
    results = pipeline(
        dataset="drugcombdb",
        model=model,
        batch_size=5120,
        epochs=100,
    )
    print(f"AUROC : {results.roc_auc:.4f}")


if __name__ == '__main__':
    main()
